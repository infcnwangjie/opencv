# -*- coding: utf-8 -*-
# from gevent import monkey;
import os
import pickle

import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, ROIS_DIR, LEFT_MARK_FROM, LEFT_MARK_TO, RIGHT_MARK_FROM, RIGHT_MARK_TO, \
	PROGRAM_DATA_DIR
# monkey.patch_all()
from itertools import chain
from queue import Queue, LifoQueue
import cv2
import time
import gevent
import math
import profile

# 把程序变成协程的方式运行②
from app.core.processers.bag_detector import BagDetector

cv2.useOptimized()

rows, cols = IMG_HEIGHT, IMG_WIDTH

SLIDE_WIDTH = 25
SLIDE_HEIGHT = 25

FOND_RECT_WIDTH = 70
FOND_RECT_HEIGHT = 70

LEFT_START = 150
LEFT_END = 175

RIGHT_START = 766
RIGHT_END = 796

tasks = Queue()
good_rects = []
results = Queue()
step = 2
fail_time = 0


def tjtime(fun):
	def inner(*args, **kwargs):
		start = time.clock()
		result = fun(*args, **kwargs)
		end = time.clock()
		print("{}cost {}秒".format(fun.__name__, end - start))
		return result

	return inner


class NearLandMark:
	__slots__ = ['col', 'row', '_slide_img', '_similarity', '_roi', 'direct']

	def __init__(self, col, row, slide_img, similarity=0):
		self.col = col
		self.row = row
		self._slide_img = slide_img
		self._similarity = similarity
		self._roi = None
		self.direct = 'left' if self.col < 0.5 * cols else 'right'  # 0 :L 1:R

	@property
	def data(self):
		return self.col, self.row, self._slide_img

	@property
	def similarity(self):
		return self._similarity

	@similarity.setter
	def similarity(self, value):
		self._similarity = value

	@property
	def roi(self):
		return self._roi

	@roi.setter
	def roi(self, value):
		value.times += 1
		self._roi = value


class TargetRect:
	__slots__ = ['point1', 'point2']

	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	def slider_in_rect(self, slide_obj: NearLandMark = None, slide_col=None, slide_row=None):
		if slide_obj:
			slide_col, slide_row, _img = slide_obj.data
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		else:
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		if self.point1[0] < slide_point1[0] and self.point1[1] < slide_point1[1] and self.point2[0] > slide_point2[
			0] and self.point2[1] > slide_point2[1]:
			return True
		return False


class LandMarkRoi:
	def __init__(self, img, label, id=None):
		self.roi = img
		self.id = id
		self.label = label
		self._times = 0
		self.land_marks = []

	def add_slide_window(self, slide_window: NearLandMark):
		# with self.lock:
		if len(self.land_marks) == 0:
			self.land_marks.append(slide_window)
		for land_mark in self.land_marks:
			col, row, similar = land_mark.col, land_mark.row, land_mark.similarity
			col1, row1, similar1 = slide_window.col, slide_window.row, slide_window.similarity
			if math.sqrt(math.pow(col - col1, 2) + math.pow(row - row1, 2)) < 50:
				if similar < similar1:
					del land_mark
				break
		else:
			self.land_marks.append(slide_window)


	@property
	def times(self):
		return self._times

	@times.setter
	def times(self, value):
		self._times = value

landmark_rois = [LandMarkRoi(img=cv2.imread(os.path.join(ROIS_DIR, roi_img)), label=roi_img.split('.')[0], id=1) for
                 roi_img in
                 os.listdir(ROIS_DIR)]



def compare_similar(img1, img2):
	if img1 is None or img2 is None:
		return 0
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
	return degree


def generator_slidewindows(dest=None):
	global rows, cols, step
	row = 0
	# cols=list(chain(range(150, 175), range(766, 800)))
	x = yield
	yield x
	# IMG_WIDTH*
	while row < rows:
		for col in chain(range(LEFT_MARK_FROM, LEFT_MARK_TO), range(RIGHT_MARK_FROM, RIGHT_MARK_TO)):
			for rect in good_rects:
				if rect.slider_in_rect(slide_col=col, slide_row=row):
					break
			else:
				yield NearLandMark(col, row, dest[row:row + SLIDE_HEIGHT, col:col + SLIDE_WIDTH])
		if fail_time > 200:
			step += 1
		elif fail_time > 10000:
			step += 300
		else:
			step = 2
		row += step


# @tjtime
def computer_task(landmark_roi: LandMarkRoi, slide_window_obj):
	# while tasks.qsize()>0:
	# slide_window_obj = tasks.get()
	if slide_window_obj is None: return
	col, row, slide_img = slide_window_obj.data
	roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
	similar = compare_similar(roi, slide_img)
	global step, fail_time
	if similar > 0.56:
		slide_window_obj.similarity = similar
		slide_window_obj.roi = landmark_roi
		landmark_roi.add_slide_window(slide_window_obj)
		fail_time = 0
		good_rects.append(TargetRect((col - FOND_RECT_WIDTH,
		                              row - FOND_RECT_HEIGHT),
		                             (col + FOND_RECT_WIDTH,
		                              row + FOND_RECT_HEIGHT)))

		# if col < 0.5 * cols:
		# 	gen_slider.send(
		# 		NearLandMark(RIGHT_START, row, dest[row:row + SLIDE_HEIGHT, RIGHT_START:RIGHT_START + SLIDE_WIDTH]))
		# else:
		# 	gen_slider.send(
		# 		NearLandMark(LEFT_START, row, dest[row:row + SLIDE_HEIGHT, LEFT_START:LEFT_START + SLIDE_WIDTH]))

		return slide_window_obj
	else:
		del slide_window_obj
		fail_time += 1


# @tjtime
def start_location_landmark(img):
	start = time.clock()
	dest = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

	for slide_window_obj in generator_slidewindows(dest):
		# 迭代结束条件
		need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times == 0]
		if len(need_find_roi) == 0:
			print("need find roi is {}".format(len(need_find_roi)))
			break

		for landmark_roi in landmark_rois:
			task = gevent.spawn(computer_task, landmark_roi, slide_window_obj)
			task.join()

	position_dic = {}
	for landmark_roi in landmark_rois:
		for slide_window_obj in landmark_roi.land_marks:
			col = slide_window_obj.col
			row = slide_window_obj.row

			cv2.rectangle(dest, (col, row), (col + SLIDE_WIDTH, row + SLIDE_HEIGHT), color=(0, 255, 255),
			              thickness=1)
			position_dic[landmark_roi.label] = [col, row]
			cv2.putText(dest,
			            "{}:{}:{}".format(landmark_roi.label, slide_window_obj.direct,
			                              round(slide_window_obj.similarity, 3)),
			            (col - 50, row + 30),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

	end = time.clock()
	print("结束{}".format(end - start))
	return dest, position_dic


# 透视变化
def perspective_transform(src, position_dic):
	H_rows, W_cols = src.shape[:2]
	print(H_rows, W_cols)

	with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
		real_positions = pickle.load(coordinate)
	real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
	                     real_positions.items()}
	print(real_position_dic)

	img_left_2, img_right_2, img_left_4, img_right_4 = position_dic.get('NO2_L'), position_dic.get(
		'NO2_R'), position_dic.get('NO4_L'), position_dic.get('NO4_R')

	real_left_2, real_right_2, real_left_4, real_right_4 = real_position_dic.get('NO2_L'), real_position_dic.get(
		'NO2_R'), real_position_dic.get('NO4_L'), real_position_dic.get('NO4_R')

	# 原图中四个角点(左上、右上、左下、右下),与变换后矩阵位置
	# pts1 = np.float32([[161, 80], [449, 12], [1, 430], [480, 394]])
	pts1 = np.float32([img_left_2, img_right_2, img_left_4, img_right_4])
	pts2 = np.float32([real_left_2, real_right_2, real_left_4, real_right_4])
	# pts2 = np.float32([[0, 0], [W_cols, 0], [0, H_rows], [H_rows, W_cols] ])

	# 生成透视变换矩阵；进行透视变换
	M = cv2.getPerspectiveTransform(pts1, pts2)

	dst = cv2.warpPerspective(src, M, (W_cols, H_rows))

	return dst, real_position_dic


def draw_grid_lines(img):
	H_rows, W_cols = img.shape[:2]
	for row in range(0,H_rows):
		if row % 50==0:
			cv2.line(img,(0,row),(W_cols,row),color=(191,62,255),thickness=1,lineType=cv2.LINE_8)
	for col in range(0,W_cols):
		if col %50 ==0:
			cv2.line(img,(col,0),(col,H_rows),color=(191,62,255),thickness=1,lineType=cv2.LINE_8)

if __name__ == '__main__':
	src, position_dic = start_location_landmark(img=cv2.imread('D:/2020-04-10-15-26-22test.bmp'))
	print(position_dic)

	dest, real_position_dic = perspective_transform(src, position_dic)
	b = BagDetector(dest)
	print(b.processed_bag)

	draw_grid_lines(dest)
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

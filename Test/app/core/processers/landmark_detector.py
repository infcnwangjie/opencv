# -*- coding: utf-8 -*-
# from gevent import monkey;
import os
import pickle
import random
from functools import cmp_to_key

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
from app.log.logtool import mylog_error

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


import re


# pattern = re.compile(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''')


def landmarkname_cmp(a, b):
	result_a = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', a[0])
	a_no = int(result_a.group(1))

	result_b = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', b[0])
	b_no = int(result_b.group(1))
	if a_no > b_no:
		return 1
	else:
		return -1


def get_next_no(landmark_name):
	result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', landmark_name)

	if result is None:
		return landmark_name

	current_no = int(result.group(1))

	if current_no > 6:
		next_no = current_no - 1
	else:
		next_no = current_no + 1

	next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
	return next_landmark


def get_opposite_landmark(landmark_name):
	import re
	result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', landmark_name)

	if result is None:
		return landmark_name

	current_no = int(result.group(1))
	current_d = result.group(2)
	opposite_d = 'R' if current_d == 'L' else 'L'
	next_landmark = "NO{NO}_{D}".format(NO=current_no, D=opposite_d)
	return next_landmark


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


def spawn(dest=None):
	global rows, cols, step
	row = 0
	# cols=list(chain(range(150, 175), range(766, 800)))
	x = yield
	yield x
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


# 透视变化
def perspective_transform(src, position_dic):
	def cmp(a, b):
		return True

	H_rows, W_cols = src.shape[:2]
	# print(H_rows, W_cols)
	detected_landmarks = len(position_dic.items())

	with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
		real_positions = pickle.load(coordinate)
	real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
	                     real_positions.items()}
	# print(real_position_dic)

	if position_dic is not None and detected_landmarks >= 3:
		left_points = [(landmark_name, (x, y)) for landmark_name, (x, y) in position_dic.items() if x < 0.5 * W_cols]
		right_points = list(filter(lambda item: item[1][0] > 0.5 * W_cols, position_dic.items()))
		left_points = sorted(left_points, key=cmp_to_key(landmarkname_cmp), reverse=False)
		right_points = sorted(right_points, key=cmp_to_key(landmarkname_cmp), reverse=False)
	else:
		# 处理失败直接返回
		mylog_error("检测到的地标小于三个，无法使用")
		return src

	if len(left_points) == detected_landmarks or len(right_points) == detected_landmarks:
		mylog_error("所有的地标都在一条直线上，无法使用")
		return src

	if len(left_points) >= 2 and len(right_points) >= 2:
		print(left_points)
		print(right_points)
		((left1_landmark, [x1, y1])) = left_points[1]
		left2_landmark = get_next_no(get_next_no(left1_landmark))
		[x2, y2] = position_dic[left2_landmark]

		right1_landmark = get_opposite_landmark(left1_landmark)
		[x3, y3] = position_dic[right1_landmark]

		right2_landmark = get_opposite_landmark(left2_landmark)
		[x4, y4] = position_dic[right2_landmark]

		print(left1_landmark, left2_landmark, right1_landmark, right2_landmark)
		pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		pts2 = np.float32([real_position_dic.get(left1_landmark), real_position_dic.get(left2_landmark),
		                   real_position_dic.get(right1_landmark), real_position_dic.get(right2_landmark)])

		# 生成透视变换矩阵；进行透视变换
		M = cv2.getPerspectiveTransform(pts1, pts2)

		dst = cv2.warpPerspective(src, M, (W_cols, H_rows))
		return dst

	elif len(left_points) >= len(right_points):
		landmark_r1, [x1, y1] = right_points[0]
		landmark_r2 = get_next_no(landmark_r1)
		x2 = x1
		landmark_L2 = get_opposite_landmark(landmark_r2)
		landmark_L2, [x3, y3] = position_dic[landmark_L2]
		y2 = y3
		landmark_L1 = get_opposite_landmark(landmark_r1)
		landmark_L1, [x4, y4] = position_dic[landmark_L1]

		pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		pts2 = np.float32([real_position_dic.get(landmark_r1), real_position_dic.get(landmark_r2),
		                   real_position_dic.get(landmark_L2), real_position_dic.get(landmark_L1)])

		# 生成透视变换矩阵；进行透视变换
		M = cv2.getPerspectiveTransform(pts1, pts2)

		dst = cv2.warpPerspective(src, M, (W_cols, H_rows))
		return dst

	elif len(left_points) <= len(right_points):
		landmark_L1, [x1, y1] = left_points[0]
		landmark_L2 = get_next_no(landmark_L1)
		x2 = x1
		landmark_R2 = get_opposite_landmark(landmark_L2)
		landmark_R2, [x3, y3] = position_dic[landmark_R2]
		y2 = y3
		landmark_R1 = get_opposite_landmark(landmark_L1)
		landmark_R1, [x4, y4] = position_dic[landmark_R1]
		pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		pts2 = np.float32([real_position_dic.get(landmark_L1), real_position_dic.get(landmark_L2),
		                   real_position_dic.get(landmark_R2), real_position_dic.get(landmark_R1)])

		# 生成透视变换矩阵；进行透视变换
		M = cv2.getPerspectiveTransform(pts1, pts2)

		dst = cv2.warpPerspective(src, M, (W_cols, H_rows))
		return dst



# @tjtime
def location_landmark(img):
	start = time.clock()
	dest = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

	for slide_window_obj in spawn(dest):
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
	dest = perspective_transform(dest, position_dic)
	return dest


def draw_grid_lines(img):
	H_rows, W_cols = img.shape[:2]
	for row in range(0, H_rows):
		if row % 50 == 0:
			cv2.line(img, (0, row), (W_cols, row), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)
	for col in range(0, W_cols):
		if col % 50 == 0:
			cv2.line(img, (col, 0), (col, H_rows), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)


if __name__ == '__main__':
	src = location_landmark(img=cv2.imread('D:/2020-04-10-15-26-22test.bmp'))

	b = BagDetector(src)
	print(b.processed_bag)

	draw_grid_lines(src)
	cv2.namedWindow("dest")
	cv2.imshow("dest", src)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

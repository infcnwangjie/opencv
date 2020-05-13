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
from app.core.location.models import LandMarkRoi, NearLandMark, TargetRect
from app.core.processers.bag_detector import BagDetector
from app.log.logtool import mylog_error
import re

cv2.useOptimized()

rows, cols = IMG_HEIGHT, IMG_WIDTH

SLIDE_WIDTH, SLIDE_HEIGHT = 24, 26

FOND_RECT_WIDTH, FOND_RECT_HEIGHT = 50, 50

LEFT_START, LEFT_END = 150, 175

RIGHT_START, RIGHT_END = 766, 796
# RIGHT_START, RIGHT_END = 490, 540
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


class LandMarkDetecotr:

	def __init__(self, img):
		self.img = img

	def position_landmark(self):
		start = time.clock()
		dest = cv2.resize(self.img, (IMG_WIDTH, IMG_HEIGHT))
		landmark_rois = self.__get_landmark_rois()
		for slide_window_obj in self.__spawn(dest):
			# 迭代结束条件
			need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times == 0]
			if len(need_find_roi) == 0:
				print("need find roi is {}".format(len(need_find_roi)))
				break

			for landmark_roi in landmark_rois:
				task = gevent.spawn(self.__check_slide_window, landmark_roi, slide_window_obj)
				task.join()

		position_dic = {}
		for landmark_roi in landmark_rois:
			landmark = landmark_roi.landmark
			if landmark is None:
				continue
			col = landmark.col
			row = landmark.row

			cv2.rectangle(dest, (col, row), (col + SLIDE_WIDTH, row + SLIDE_HEIGHT), color=(0, 255, 255),
			              thickness=1)
			position_dic[landmark_roi.label] = [col, row]
			cv2.putText(dest,
			            "{}:{}:{}".format(landmark_roi.label, landmark.direct,
			                              round(landmark.similarity, 3)),
			            (col - 50, row + 30),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

		end = time.clock()
		print("结束{}".format(end - start))
		dest = self.__perspective_transform(dest, position_dic)
		# self.__draw_grid_lines(dest)
		return dest

	def __get_landmark_rois(self):
		# landmark_rois = [LandMarkRoi(img=cv2.imread(os.path.join(ROIS_DIR, roi_img)), label=roi_img.split('.')[0], id=1)
		#                  for
		#                  roi_img in
		#                  os.listdir(ROIS_DIR)]
		landmark_rois = [LandMarkRoi(img=cv2.imread("d:/T_G_R_.png"), label="T_G_R_", id=1),
		                 LandMarkRoi(img=cv2.imread("d:/NO_R.png"), label="red", id=2)]
		return landmark_rois

	def __draw_grid_lines(self, img):
		H_rows, W_cols = img.shape[:2]
		for row in range(0, H_rows):
			if row % 50 == 0:
				cv2.line(img, (0, row), (W_cols, row), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)
		for col in range(0, W_cols):
			if col % 50 == 0:
				cv2.line(img, (col, 0), (col, H_rows), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)

	def __perspective_transform(self, src, position_dic):
		'''透视变化'''
		H_rows, W_cols = src.shape[:2]
		# print(H_rows, W_cols)
		detected_landmarks = len(position_dic.items())

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}
		print(real_position_dic)

		if position_dic is not None and detected_landmarks >= 3:
			left_points = [(landmark_name, (x, y)) for landmark_name, (x, y) in position_dic.items() if
			               x < 0.5 * W_cols]
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

	def __compare_similar(self, img1, img2):
		if img1 is None or img2 is None:
			return 0

		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
		return degree

	def __check_slide_window(self, landmark_roi: LandMarkRoi, slide_window_obj):
		if slide_window_obj is None: return
		col, row, slide_img = slide_window_obj.data
		roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
		similar = self.__compare_similar(roi, slide_img)
		# print(similar,col,row)

		global step, fail_time
		if similar > 0.51:
			slide_window_obj.similarity = similar
			slide_window_obj.roi = landmark_roi
			landmark_roi.add_slide_window(slide_window_obj)
			fail_time = 0
			good_rects.append(TargetRect((col - FOND_RECT_WIDTH,
			                              row - FOND_RECT_HEIGHT),
			                             (col + FOND_RECT_WIDTH,
			                              row + FOND_RECT_HEIGHT)))
			return slide_window_obj
		else:
			del slide_window_obj
			fail_time += 1

	def __landmark_position_dic(self):
		'''获取所有的地标标定位置'''
		with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
			real_positions = pickle.load(coordinate)
		return real_positions

	def __spawn(self, dest=None):
		global rows, cols, step
		row = 0
		# cols=list(chain(range(150, 175), range(766, 800)))
		x = yield
		yield x
		while row < rows:
			# for col in chain(range(LEFT_MARK_FROM, LEFT_MARK_TO), range(RIGHT_MARK_FROM, RIGHT_MARK_TO)):
			for col in range(660, 690):
				# for col in chain(range(LEFT_MARK_FROM, LEFT_MARK_TO), range(660,690)):
				for rect in good_rects:
					if rect.slider_in_rect(slide_col=col, slide_row=row):
						break
				else:
					# cv2.rectangle(dest, (col, row), (col + SLIDE_WIDTH, row + SLIDE_HEIGHT), color=(0, 255, 255),
					#               thickness=1)
					yield NearLandMark(col, row, dest[row:row + SLIDE_HEIGHT, col:col + SLIDE_WIDTH])
			if fail_time > 100:
				step += 1
			elif fail_time > 1000:
				step += 50
			else:
				step = 1
			row += step


def test1():
	# src = LandMarkDetecotr(img=cv2.imread('D:/2020-04-10-15-26-22test.bmp')).position_remark()
	src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-12-10-53-30test.bmp')).position_landmark()  # 0.72
	# src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-12-10-52-56test.bmp')).position_remark() #0.916
	# src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-12-10-52-56test.bmp')).position_remark()  # 0.916
	b = BagDetector(src)
	print(b.location_bag())
	# __draw_grid_lines(src)
	cv2.namedWindow("dest")
	cv2.imshow("dest", src)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test2():
	def compute_hist(img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hist = cv2.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		# hist = cv2.calcHist([img], [0], None, [256], [0, 255.0])
		# cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		return hist

	# def __compare_similar(img1, img2):
	# 	if img1 is None or img2 is None:
	# 		return 0
	#
	# 	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
	# 	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
	# 	hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255.0])
	# 	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	# 	hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255.0])
	# 	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	# 	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
	# 	return degree

	img = cv2.imread('d:/2020-05-12-10-53-30test.bmp')
	img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
	hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	cv2.namedWindow("img")
	cv2.imshow("img", img)
	img_roi = cv2.imread("d:/T_G_R_.png")
	img_roi = cv2.resize(img_roi, (40, 40))
	img_roi_hsvt = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
	cv2.namedWindow("img_roi")
	cv2.imshow("img_roi",img_roi)
	# landmarkroi = LandMarkRoi(img=cv2.imread("d:/T_G_R_.png"), label="T_G_R_", id=1)
	roihist=compute_hist(img_roi_hsvt)
	cv2.namedWindow("roihist")
	cv2.imshow("roihist",roihist)
	dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
	cv2.namedWindow("back")
	cv2.imshow("back",dst)
	cv2.waitKey(0)

def test3():
	import cv2
	import numpy as np

	# 目标搜索图片
	target = cv2.imread('d:/2020-05-12-10-53-30test.bmp')
	target = cv2.resize(target, (IMG_WIDTH, IMG_HEIGHT))
	hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)


	# roi图片，就想要找的的图片
	roi = cv2.imread('d:/T_G_R_.png')
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


	# 计算目标直方图
	roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
	# 归一化，参数为原图像和输出图像，归一化后值全部在2到255范围
	cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
	dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

	# 卷积连接分散的点
	disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	dst = cv2.filter2D(dst, -1, disc)

	ret, thresh = cv2.threshold(dst, 50, 255, 0)
	# 使用merge变成通道图像
	# thresh = cv2.merge((thresh, thresh, thresh))
	thresh = cv2.medianBlur(thresh, 3)

	contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		area=cv2.contourArea(contour)
		if area<100:continue
		rect = cv2.boundingRect(contour)
		rect_x, rect_y, rect_w, rect_h = rect
		cv2.rectangle(target, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color=(0, 255, 255),
		              thickness=1)

	# 蒙板
	# res = cv2.bitwise_and(target, thresh)
	# 矩阵按列合并,就是把target,thresh和res三个图片横着拼在一起
	# cv2.imwrite('res.jpg', res)
	# 显示图像
	cv2.imshow('1', thresh)
	cv2.imshow('target', target)
	cv2.waitKey(0)




if __name__ == '__main__':
	# test1()
	test3()
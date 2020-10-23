from time import sleep, time
import profile
import cv2
import numpy as np
import re
import ctypes
from ctypes import cdll
from collections import defaultdict
from functools import cmp_to_key, reduce, partial

# from aip import AipOcr

from app.core.beans.models import *
from app.core.support.shapedetect import ShapeDetector, Shape
from app.core.video.imageprovider import ImageProvider
from app.core.video.mvs.MvCameraSuppl_class import MvSuply
from app.core.autowork.detector import LandMarkDetecotr, BagDetector, BaseDetector, LasterDetector, HockDetector, \
	shrink_coach_area, logger
from app.core.video.sdk import SdkHandle


def test_match():
	import cv2
	import numpy as np
	cap = cv2.VideoCapture("D:\\Video_20200725072828533.avi")  # Video_20200725072828533.avi

	# cap = cv2.VideoCapture("D:\\Video_20200724124902730.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()
	landmark_detect = LandMarkDetecotr()
	hock_detect = HockDetector()

	# cv2.namedWindow("show")
	# cv2.namedWindow("foreground")
	cv2.namedWindow("orbTest", 0)

	img1 = cv2.imread("D:/NTY_IMG_PROCESS/BAG_ROI/bag.png")  # 导入灰度图像
	detector = cv2.ORB_create()
	# kp1 = detector.detectAndCompute(img1, None)
	kp1, des1 = detector.detectAndCompute(img1, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	index = 0

	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		rows, cols, channels = show.shape
		if rows != 700 or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		img2 = show
		# index+=1
		# if index<200:
		#     continue
		# if index>201:
		#     break
		kp2, des2 = detector.detectAndCompute(img2, None)

		# kp2 = detector.detect(img2, None)
		#
		# kp2, des2 = detector.compute(img2, kp2)

		# matches = bf.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# matches = bf.knnMatch(des1, des2,k=1)
		matches = bf.match(des1, des2)
		# matches = sorted(matches, key=lambda x: x.distance)  # 绘制前10的匹配项
		for matchitem in matches:
			print(matchitem)
		# cv2.imshow("t",des2)
		matches = sorted(matches, key=lambda x: x.distance)  # 据距离来排序
		# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
		img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:30], img2, flags=2)

		cv2.imshow('orbTest', img3)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def filter_matches(kp1, kp2, matches, ratio=0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append(kp1[m.queryIdx])
			mkp2.append(kp2[m.trainIdx])
	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])
	kp_pairs = zip(mkp1, mkp2)
	return p1, p2, kp_pairs


def test_hock():
	# TODO https://blog.csdn.net/ikerpeng/article/details/47972959
	import cv2
	import numpy as np
	cap = cv2.VideoCapture("H:\\Video_20200831092239170.avi")  # Video_20200725072828533.avi

	# cap = cv2.VideoCapture("H:\\Video_20200827091912396.avi")#Video_20200725072828533.avi
	# cap = SdkHandle()
	landmark_detect = LandMarkDetecotr()
	hock_detect = HockDetector()

	# cv2.namedWindow("show")
	# cv2.namedWindow("foreground")
	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			continue
		rows, cols, channels = show.shape
		if rows != 700 or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		hsv = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)

		green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		# 去除颜色范围外的其余颜色
		green_mask = cv2.inRange(hsv, green_min, green_max)
		ret, green_binary = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)
		green_binary = cv2.resize(green_binary, (IMG_HEIGHT, IMG_WIDTH))
		# img=np.concatenate((show,green_binary))


		cv2.imshow("green_binary", green_binary)
		cv2.imshow("show", show)



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


def read_config():
	import os
	import re
	config_pattern = re.compile("([A-Za-z_0-9]+)\s*\=\s*(.*)")

	try:
		with open("config.txt", 'rt', encoding="utf-8") as config_handle:
			for line in config_handle.readlines():
				if len(line) == 0 or line == "":
					continue
				match_result = re.match(config_pattern, line)
				if match_result is not None:
					key = match_result.group(1)
					value = match_result.group(2)
					print("key:{},value:{}".format(key, value))
	except Exception as e:
		raise e


# -*- coding:utf-8 -*-
__author__ = 'Microcosm'

import cv2
# from find_obj import filter_matches,explore_match
import numpy as np


def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1 + w2] = img2
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	if H is not None:
		corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
		corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
		cv2.polylines(vis, [corners], True, (255, 255, 255))

	if status is None:
		status = np.ones(len(list(kp_pairs)), np.bool)
	p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	green = (0, 255, 0)
	red = (0, 0, 255)
	white = (255, 255, 255)
	kp_color = (51, 103, 236)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			col = green
			cv2.circle(vis, (x1, y1), 2, col, -1)
			cv2.circle(vis, (x2, y2), 2, col, -1)
		else:
			col = red
			r = 2
			thickness = 3
			cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
			cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
			cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
			cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
	vis0 = vis.copy()
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			cv2.line(vis, (x1, y1), (x2, y2), green)

	cv2.imshow(win, vis)


def test_single_match():
	img1 = cv2.imread("D:/big.png")

	img1 = cv2.resize(img1, (900, 700))

	img2 = cv2.imread("D:/NTY_IMG_PROCESS/SUPPORT_ROI/hock.png")
	#
	# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	sift = cv2.ORB_create()

	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# BFmatcher with default parms
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg=img1, flags=2)
	cv2.imshow("img3", img3)

	# p1, p2, kp_pairs = filter_matches(kp1, kp2, matches, ratio=0.5)
	# explore_match('matches', img1_gray, img2_gray, kp_pairs)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test_landmark_bag():
	import cv2
	import numpy as np
	show = None

	cv2.namedWindow("show")

	# cv2.namedWindow("foreground")
	def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
		global show
		if event == cv2.EVENT_LBUTTONDOWN:
			xy = "% d, % d" % (x, y)
			print('x, y = {},{}'.format(x, y))

	# TODO https://blog.csdn.net/ikerpeng/article/details/47972959

	loc = cv2.setMouseCallback("show", on_EVENT_LBUTTONDOWN)

	cap = cv2.VideoCapture("D:/Video_20201023084819986.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()
	# cap = cv2.VideoCapture("D:\\Video_20200825135910294.avi")  # Video_20200725072828533.avi
	landmark_detect = LandMarkDetecotr()
	hock_detect = HockDetector()
	bag_detect = BagDetector()

	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			break
		rows, cols, channels = show.shape
		rows, cols, channels = show.shape
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		perspective_img, find_landmark = landmark_detect.position_landmark(show)
		perspective_img_copy = perspective_img.copy()
		if find_landmark:
			# ignore, foregroud = hock_detect.location_hock_withlandmark(perspective_img, perspective_img_copy,
			#                                                            find_landmark,
			#                                                            middle_start=110,
			#                                                            middle_end=508)

			# cv2.drawContours(perspective_img,contours)
			foreground, contours = bag_detect.red_contours(perspective_img,110, 508)

			ret, foreground = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)

			try:
				foreground_bagarea = shrink_coach_area(perspective_img)
				cv2.imshow("foreground_bagarea", foreground_bagarea)
			except Exception as e:
				logger(e.__str__(), "error")
			else:
				foreground = cv2.bitwise_and(foreground, foreground, mask=foreground_bagarea)
			bagcontours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# cv2.imshow('bag_img', binary_img)
			cv2.drawContours(perspective_img, bagcontours, -1, (0, 0, 255), 3)
			cv2.imshow('show', show)
			cv2.imshow('perspective_img', perspective_img)
		# cv2.imshow("foregroud", foregroud)
		else:
			# hock, foregroud = hock_detect.location_hock_withoutlandmark(show,middle_start=237,
			#                           middle_end=502)
			# if hock is not None:
			#     print("hock_w:{},hock_h:{}".format(hock.w, hock.h))
			cv2.imshow("show", show)
		# cv2.imshow("foregroud", foregroud)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


def test_roi_match():
	import cv2
	import numpy as np

	cv2.namedWindow("show")
	landmark_detect = LandMarkDetecotr()
	cap = cv2.VideoCapture("H:\\Video_20200831092239170.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()

	while True:
		# sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			break
		rows, cols, channels = show.shape
		if rows != 700 or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		show = cv2.resize(show, (700, 900))

		perspective_img, find_landmark = landmark_detect.position_landmark(show)

		# foreground = MvSuply.FIND_IT(show, cv2.imread("D:/NTY_IMG_PROCESS/ROIS/NO5_L.png"))

		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.filter2D(foreground, -1, disc)
		# cv2.imshow("foreground", foreground)
		cv2.imshow("show", perspective_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


# foreground = MvSuply.FIND_IT(cv2.imread("D:/test_yg.jpg"), cv2.imread("D:/NTY_IMG_PROCESS/ROIS/NO3_L.png"))
# cv2.imshow("foreground",foreground)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 用来判断区域,方法得封装起来
def area_detect():
	global LANDMARK_COLOR_INFO, COLOR_RANGE

	COLOR_INPUT = ['RED', 'GREEN', 'BLUE']

	LANDMARK_COLOR_INFO = {'NO1_L_COLOR': 'GREEN', 'NO2_L_COLOR': 'RED', 'NO3_L_COLOR': 'BLUE', 'NO4_L_COLOR': 'RED',
	                       'NO5_L_COLOR': 'GREEN', 'NO6_L_COLOR': 'BLUE'}

	left_landmarks = list(filter(lambda color: 'L' in color[0][0:5], LANDMARK_COLOR_INFO.items()))
	left_landmarks = sorted(left_landmarks, key=lambda item: item[0][2], reverse=False)
	# print(left_landmarks)

	start = 0
	choosed_landmarks = []
	find = False
	while start <= len(left_landmarks):
		start += 1
		choosed_landmarks = [item for item in left_landmarks[start:]]
		if len(choosed_landmarks) < 3: break
		# print(choosed_landmarks[0:3])

		for index in range(3):
			if COLOR_INPUT[index] != choosed_landmarks[0:3][index][1]:
				# print(choosed_landmarks[0:3][index][1])
				break
		else:
			print("ok,find it ,first:{},second:{},third:{}".format(choosed_landmarks[0][0][0:5],
			                                                       choosed_landmarks[1][0][0:5],
			                                                       choosed_landmarks[2][0][0:5]))
			find = True

		if find == True:
			break

	return choosed_landmarks


def test_sort():
	list1 = [9, 10, 1, 8, 5]
	list2 = sorted(list1, key=lambda c: c, reverse=True)
	print(list2)


def mytest():
	img = cv2.imread("D:/Image_20200827091204330.bmp")
	show = cv2.resize(img, (700, 800))
	hsv = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)

	green_low, green_high = [35, 43, 46], [77, 255, 255]
	green_min, green_max = np.array(green_low), np.array(green_high)
	# 去除颜色范围外的其余颜色
	green_mask = cv2.inRange(hsv, green_min, green_max)
	ret, green_binary = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)
	green_binary = cv2.resize(green_binary, (700, 800))

	red_low, red_high = [156, 43, 46], [180, 255, 255]
	red_min, red_max = np.array(red_low), np.array(red_high)
	# 去除颜色范围外的其余颜色
	red_mask = cv2.inRange(hsv, red_min, red_max)
	ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
	red_binary = cv2.resize(red_binary, (700, 800))

	yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]
	yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)
	# 去除颜色范围外的其余颜色
	yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)
	ret, yellow_binary = cv2.threshold(yellow_mask, 0, 255, cv2.THRESH_BINARY)
	yellow_binary = cv2.resize(yellow_binary, (700, 800))

	# HEJI_IMG = np.vstack([green_binary, red_binary, yellow_binary])
	# HEJI_IMG = cv2.resize(HEJI_IMG, (700, 800))
	# cv2.imshow("HEJI_IMG", HEJI_IMG)
	# cv2.imshow("green", red_binary)

	blue_low, bluegreen_high = [100, 43, 46], [124, 255, 255]
	blue_min, blue_max = np.array(blue_low), np.array(bluegreen_high)
	# 去除颜色范围外的其余颜色
	blue_mask = cv2.inRange(hsv, blue_min, blue_max)
	ret, blue_binary = cv2.threshold(blue_mask, 0, 255, cv2.THRESH_BINARY)
	blue_binary = cv2.resize(blue_binary, (700, 800))
	left_open_mask = np.zeros_like(blue_binary)
	left_open_mask[0:IMG_HEIGHT, 170:187] = 255

	right_open_mask = np.zeros_like(blue_binary)
	right_open_mask[0:IMG_HEIGHT, 547:564] = 255

	blue_binary = cv2.bitwise_and(blue_binary, blue_binary, mask=left_open_mask)
	green_binary = cv2.bitwise_and(green_binary, green_binary, mask=left_open_mask)
	red_binary = cv2.bitwise_and(red_binary, red_binary, mask=left_open_mask)

	cv2.imshow("green", green_binary)
	cv2.imshow("red", red_binary)
	# cv2.imshow("yellow",yellow_binary)
	cv2.imshow("blue_binary", blue_binary)

	# find_it_foreground = MvSuply.FIND_IT(show, cv2.imread("D:/NTY_IMG_PROCESS/ROIS/NO3_L.png"))
	#
	# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# find_it_foreground = cv2.filter2D(find_it_foreground, -1, disc)
	# cv2.imshow("find_it_foreground", find_it_foreground)
	cv2.imshow("show", show)
	cv2.waitKey(0)


def check_bag_area():
	def inner_filter(c):
		x, y, w, h = cv2.boundingRect(c)
		if w < 100 or h < 100:
			return False
		if cv2.contourArea(c) < 10000:
			return False
		return True

	import cv2
	import numpy as np

	cv2.namedWindow("show")
	cap = cv2.VideoCapture("D:\\Video_20200831092239170.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()
	landmark_detect = LandMarkDetecotr()

	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			break
		rows, cols, channels = show.shape
		if rows != 700 or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		show, find_landmark = landmark_detect.position_landmark(show)

		# hsv = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)
		# white_low, white_high = [0, 0, 221], [180, 30, 255]
		# white_min, white_max = np.array(white_low), np.array(white_high)
		# # 去除颜色范围外的其余颜色
		# white_mask = cv2.inRange(hsv, white_min, white_max)
		# ret, white_binary = cv2.threshold(white_mask, 0, 255, cv2.THRESH_BINARY)
		# white_binary = cv2.resize(white_binary, (700, 800))

		gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)

		ret, white_binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		foreground = cv2.dilate(white_binary, kernel)

		if find_landmark == True:
			middle_open_mask = np.zeros_like(gray)
			middle_open_mask[0:rows, 110:508] = 255
		else:
			middle_open_mask = np.zeros_like(gray)
			middle_open_mask[0:rows, 228:510] = 255

		foreground = cv2.bitwise_and(foreground, foreground, mask=middle_open_mask)

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		contours = sorted(list(filter(lambda c: inner_filter(c), contours)), key=lambda c: cv2.contourArea(c),
		                  reverse=True)

		contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
		# contours = sorted(list(filter(lambda c: inner_filter(c), contours)), key=lambda c: cv2.contourArea(c),
		#                   reverse=True)
		#
		# cv2.drawContours(perspect_img, contours, -1, (255, 255, 255), 3)
		black_empty = np.zeros_like(gray)
		# cv2.drawContours(black_empty,[contours[0]],-1,(255,255,255),1)
		cv2.fillPoly(black_empty, [contours[0]], (255, 255, 255))

		# foreground = MvSuply.FIND_IT(show, cv2.imread("D:/NTY_IMG_PROCESS/ROIS/NO1_L.png"))

		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.filter2D(white_binary, -1, disc)
		cv2.imshow("black_empty", black_empty)
		cv2.imshow("show", show)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


# def test_tensorflow():
	# import tensorflow as tf

	# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
	# 加到默认图中.
	#
	# 构造器的返回值代表该常量 op 的返回值.
	# matrix1 = tf.constant([[3., 3.]])

	# 创建另外一个常量 op, 产生一个 2x1 矩阵.
	# matrix2 = tf.constant([[2.], [2.]])

	# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
	# 返回值 'product' 代表矩阵乘法的结果.
	# product = tf.matmul(matrix1, matrix2)
	# print(product)


def test_category():
	# Image_20200621160435205.bmp
	# 20200622_121041_2_0.bmp
	result = MvSuply.CATEGORY_CODE(cv2.imread("H:/NTY_IMG_PROCESS/ROIS/NO6_R.png"))
	print(result)


# def change_background():
# 	img=cv2.imread("D:/car.jpg")
# 	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
# 	color1_min, color1_max = np.array(color_low), np.array(color_high)
# 	color1_mask = cv2.inRange(hsv, color1_min, color1_max)
# 	ret, foreground = cv2.threshold(color1_mask, 0, 255, cv2.THRESH_BINARY)
# 	contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# 	edge_output =cv2.Canny(gray, 50, 150)
# 	cv2.imshow("img",edge_output)
# 	cv2.waitKey(0)

def check_placementarea():
	img = cv2.imread("D:/dddd.png")
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	color_low, color_high = [11, 43, 46], [34, 255, 255]
	color1_min, color1_max = np.array(color_low), np.array(color_high)
	color1_mask = cv2.inRange(hsv, color1_min, color1_max)
	ret, foreground = cv2.threshold(color1_mask, 0, 255, cv2.THRESH_BINARY)
	contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	find = contours is not None and len(contours) > 0 and ShapeDetector().detect(contours[0], Shape.RECTANGLE)

	# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

	cv2.putText(img, "{}".format(find), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
	            (255, 100, 255), 2)
	cv2.imshow("img", img)
	cv2.waitKey(0)


def putbagarea_test():
	LP1 = (120, 100)
	LP2 = (120, 400)
	RP1 = (470, 100)
	RP2 = (470, 400)


def test_split_x_y():
	strinfo = "(123,343)"
	t = re.compile("\((.*?),(.*?)\)")
	result = re.match(t, strinfo)
	print(result.group(1), result.group(2))


def test_find_alllandmark():
	import cv2
	import numpy as np

	cv2.namedWindow("show")
	# landmark_detect = LandMarkDetecotr()
	cap = cv2.VideoCapture("D:\\Video_20200925150759752.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()

	def is_landmark(show,c):
		x, y, w, h = cv2.boundingRect(c)
		# print(w, h)
		# if len(ws) < 3 or len(hs) < 3:
		start=time.time()
		category = MvSuply.CATEGORY_CODE(show[y-4:y + h+4, x-4:x + w+4, :])
		end=time.time()
		# print("cost:{}".format(end-start))
		if category == 0: return False
		return True

	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			break
		rows, cols, channels = show.shape
		if rows != 700 or cols != IMG_WIDTH:
			show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		else:
			show = show

		show = cv2.resize(show, (700, 900))

		# perspective_img, find_landmark = landmark_detect.position_landmark(show)
		gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)

		value=np.average(gray)
		print(value)

		# foreground = MvSuply.FIND_IT(show, cv2.imread("D:/NTY_IMG_PROCESS/ROIS/NO5_L.png"))
		ret, foreground = cv2.threshold(gray, 70, 100,
		                                cv2.THRESH_BINARY)  # 110,255

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL,
		                                        cv2.CHAIN_APPROX_SIMPLE)

		contours=list(filter(lambda c:is_landmark(show,c),contours))
		cv2.drawContours(show, contours, -1, (255, 0, 0), 3)



		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.filter2D(foreground, -1, disc)
		cv2.imshow("foreground", foreground)
		cv2.imshow("show", show)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# test_hock()
	# test_match()
	# test_single_match()
	# profile.run("test_landmark()")
	# test_sort()
	# test_roi_match()
	# mytest()
	# print(area_detect())
	# check_bag_area()
	# test_tensorflow()
	# test_category()
	# change_background()
	# check_placementarea()
	# test_landmark()
	# test_category()
	# test_split_x_y()
	# test_find_alllandmark()
	test_landmark_bag()


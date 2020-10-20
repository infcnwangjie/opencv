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
from app.config import middle_start_withlandmark
from app.core.autowork.detector import LandMarkDetecotr
from app.core.beans.models import *
from app.core.support.shapedetect import ShapeDetector, Shape
from app.core.video.imageprovider import ImageProvider
from app.core.video.mvs.MvCameraSuppl_class import MvSuply, middle_end_withlandmark
# from app.core.autowork.detector import LandMarkDetecotr, BagDetector, BaseDetector, LasterDetector, HockDetector, \
# 	shrink_coach_area, logger
from app.core.video.sdk import SdkHandle

import math


def test_hock():
	# TODO https://blog.csdn.net/ikerpeng/article/details/47972959
	import cv2
	import numpy as np

	middle_start, middle_end = 120, 470

	def filter_laster_contour(c):
		x, y, w, h = cv2.boundingRect(c)
		area = cv2.contourArea(c)
		# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
		print("laster is {}".format(area))



		if w < 20 or h < 20:
			return False

		if x < middle_start or x > middle_end:
			return False

		return True

	cap = cv2.VideoCapture("D:\\Video_20201019131548107.avi")  # Video_20200725072828533.avi

	candidate_hock_contours = []
	# import app.core.autowork.detector.LandMarkDetecotr
	landmark_detect = LandMarkDetecotr()

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

		# perspective_img, find_landmark = landmark_detect.position_landmark(show)

		# if find_landmark:
			foreground, green_contours = hock_hocks_bycolor(show)

			# foreground, green_contours=find_move_foregrond_method(show)
			# foreground, green_contours=background_sub(show)
			# foreground,green_contours=white_object(show)

			laster_contours = list(filter(filter_laster_contour, green_contours))
			laster_contours = sorted(laster_contours, key=lambda c: cv2.contourArea(c), reverse=False)
			biggest_c = None
			# hock_candidates=[]
			if laster_contours is not None and len(laster_contours) > 0:
				if len(laster_contours) > 1:
					biggest_c = laster_contours[0]
					max_area = 0
					for index, c in enumerate(laster_contours):
						area = cv2.contourArea(c)
						if area > max_area:
							max_area = area
							biggest_c = c

					b_x, b_y, b_w, b_h = cv2.boundingRect(biggest_c)
					M_biggest = cv2.moments(biggest_c)
					biggest_cx = int(M_biggest['m01'] / M_biggest['m00'])
					biggest_cy = int(M_biggest['m01'] / M_biggest['m00'])

					x_list, y_list = [], []
					for c_iter in laster_contours:
						m = cv2.moments(biggest_c)
						cx = int(m['m01'] / m['m00'])
						cy = int(m['m01'] / m['m00'])
						if abs(cx - biggest_cx) > 100 or abs(cy - biggest_cy) > 100: continue
						x, y, w, h = cv2.boundingRect(c)
						x_list.append(x)
						x_list.append(x + w)
						y_list.append(y)
						y_list.append(y + h)

					min_x, min_y = min(x_list), min(y_list)
					max_x, max_y = max(x_list), max(y_list)
					cent_x = int(0.5 * (min_x + max_x))
					cent_y = int(0.5 * (min_y + max_y))
					cv2.drawContours(show, [biggest_c], -1, (255, 0, 0), 3)
					cv2.rectangle(show, (min_x, min_y), (max_x, max_y),
					              (30, 144, 255), 3)
				# laster_contours.pop(index)
				# hock_candidates.append(biggest_c)

				# cv2.drawContours(img_show, [biggest_c], -1, (255, 0, 0), 3)
				# for c in laster_contours:
				# 	# if id(c) == id(biggest_c): continue
				# 	M = cv2.moments(c)
				# 	cx = int(M['m10'] / M['m00'])
				# 	cy = int(M['m01'] / M['m00'])
				# 	if biggest_cx is None or biggest_cy is None:
				# 		candidate_hock_contours.append(c)
				# 	elif abs(cx - biggest_cx) < 300 and abs(cy - biggest_cy) <= 300:
				# 		candidate_hock_contours.append(c)
				else:
					biggest_c = laster_contours[0]
					cv2.drawContours(show, [biggest_c], -1, (255, 0, 0), 3)




			# M_biggest = cv2.moments(biggest_c)
			# try:
			#
			# 	biggest_cx = int(M_biggest['m01'] / M_biggest['m00'])
			# 	biggest_cy = int(M_biggest['m01'] / M_biggest['m00'])
			# except :
			# 	biggest_cx=None
			# 	biggest_cy=None



			# 区域生长应该仅仅是小范围操作,不能大范围使用
			# seeds = originalSeed(foreground, th=255,contours=laster_contours)
			# foreground = regionGrow(foreground, seeds, thresh=30, p=8)
			print(candidate_hock_contours)
			# cv2.drawContours(perspective_img, laster_contours, -1, (255, 255, 255), 3)
			# point_list=[]
			# for c in laster_contours[:2]:
			# 	M=cv2.moments(c)
			# 	cx = int(M['m10'] / M['m00'])
			# 	cy = int(M['m01'] / M['m00'])
			# 	point_list.append([cx,cy])
			#
			# points = np.array([point_list])
			# rect=cv2.minAreaRect(points)
			# box = cv2.boxPoints(rect,points)  # cv2.boxPoints(rect) for OpenCV 3.x
			# box = np.int0(box)
			# cv2.drawContours(show, [box], 0, (0, 0, 255), 2)

			# img=np.concatenate((show,green_binary))
			cv2.imshow("green_binary", foreground)
			cv2.imshow("show", show)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()


def find_bigest_contour(foreground, laster_contours):
	if len(laster_contours) > 1:
		biggest_c = laster_contours[0]
		max_area = 0
		for c in laster_contours:
			area = cv2.contourArea(c)
			if area > max_area:
				max_area = area
				biggest_c = c

		x, y, w, h = cv2.boundingRect(biggest_c)
		cv2.rectangle(foreground, (x - 200, y - 200), (x + 200, y + 200),
		              (255, 0, 0), 3)

		return y - 200, y + 200, x - 200, x + 200
	return 0, 0, 0, 0


def hock_hocks_bycolor(show):
	middle_start, middle_end = 120, 470
	hsv = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)
	green_low, green_high = [35, 43, 46], [77, 255, 255]
	green_min, green_max = np.array(green_low), np.array(green_high)
	# 去除颜色范围外的其余颜色
	green_mask = cv2.inRange(hsv, green_min, green_max)
	ret, green_binary = cv2.threshold(green_mask, 20, 255, cv2.THRESH_BINARY)
	foreground = cv2.resize(green_binary, (IMG_HEIGHT, IMG_WIDTH))
	disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	foreground = cv2.filter2D(foreground, -1, disc)

	middle_mask = np.zeros_like(foreground)
	middle_mask[:, middle_start:middle_end] = 255
	foreground = cv2.bitwise_and(foreground, foreground, mask=middle_mask)
	green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return foreground, green_contours


one_frame, two_frame, three_frame = None, None, None


def find_move_foregrond_method(img):
	global one_frame, two_frame, three_frame
	frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if one_frame is None or two_frame is None or three_frame is None:
		one_frame = np.zeros_like(frame_gray)
		two_frame = np.zeros_like(frame_gray)
		three_frame = np.zeros_like(frame_gray)

	# if self.mask is None:
	# 	self.mask = np.zeros_like(frame_gray)

	one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
	# if self.one_frame.shape !=self.two_frame.shape:
	# 	return None
	abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
	_, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

	# if self.two_frame.shape !=self.three_frame.shape:
	# 	return None
	abs2 = cv2.absdiff(two_frame, three_frame)
	_, thresh2 = cv2.threshold(abs2, 40, 255, cv2.THRESH_BINARY)

	foreground = cv2.bitwise_and(thresh1, thresh2)  # 与运算
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	#
	# foreground[0:IMG_HEIGHT,0:110]=0
	# foreground[0:IMG_HEIGHT, 450:] = 0
	foreground = cv2.filter2D(foreground, -1, kernel)

	#
	# dilate = cv2.dilate(foreground, kernel)  # 膨胀
	# erode = cv2.erode(dilate, kernel)  # 腐蚀
	# dilate = cv2.dilate(dilate, kernel)  # 膨胀

	# contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
	#                                  method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

	# cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# for contour in contours:
	# 	if 100 < cv2.contourArea(contour) < 40000:
	# 		x, y, w, h = cv2.boundingRect(contour)  # 找方框
	# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

	disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	foreground = cv2.filter2D(foreground, -1, disc)
	green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return foreground, green_contours


def white_object(showimg):
	frame_gray = cv2.cvtColor(showimg, cv2.COLOR_BGR2GRAY)
	_, thresh1 = cv2.threshold(frame_gray, 200, 255, cv2.THRESH_BINARY)
	disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	foreground = cv2.filter2D(thresh1, -1, disc)
	green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return foreground, green_contours


fgbg = cv2.createBackgroundSubtractorMOG2()


def background_sub(showimg):
	global fgbg
	foreground = fgbg.apply(showimg)
	disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	foreground = cv2.filter2D(foreground, -1, disc)
	green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return foreground, green_contours


# 初始种子选择
def originalSeed(gray, th, contours):
	ret, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)  # 二值图，种子区域(不同划分可获得不同种子)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 3×3结构元
	thresh_copy = thresh.copy()  # 复制thresh_A到thresh_copy
	thresh_B = np.zeros(gray.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
	seeds = []  # 为了记录种子坐标
	# 循环，直到thresh_copy中的像素值全部为0
	while thresh_copy.any():

		Xa_copy, Ya_copy = np.where(thresh_copy > 0)  # thresh_A_copy中值为255的像素的坐标
		thresh_B[Xa_copy[0], Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255

		# 连通分量算法，先对thresh_B进行膨胀，再和thresh执行and操作（取交集）
		# for i in range(200):
		# 	dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
		# 	thresh_B = cv2.bitwise_and(thresh, dilation_B)

		# 取thresh_B值为255的像素坐标，并将thresh_copy中对应坐标像素值变为0
		Xb, Yb = np.where(thresh_B > 0)
		thresh_copy[Xb, Yb] = 0

		# 循环，在thresh_B中只有一个像素点时停止,thresh_B为缓存变量了
		# while str(thresh_B.tolist()).count("255") > 1:
		# 	thresh_B = cv2.erode(thresh_B, kernel, iterations=1)  # 腐蚀操作

		X_seed, Y_seed = np.where(thresh_B > 0)  # 取处种子坐标
		if X_seed.size > 0 and Y_seed.size > 0:
			seeds.append((X_seed[0], Y_seed[0]))  # 将种子坐标写入seeds
		thresh_B[Xb, Yb] = 0  # 将thresh_B像素值置零

	return seeds


# 区域生长
def regionGrow(gray, seeds, thresh, p):
	seedMark = np.zeros(gray.shape)
	# 八邻域
	if p == 8:
		connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
	elif p == 4:
		connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

	# seeds内无元素时候生长停止
	while len(seeds) != 0:
		# 栈顶元素出栈
		pt = seeds.pop(0)
		for i in range(p):
			tmpX = pt[0] + connection[i][0]
			tmpY = pt[1] + connection[i][1]

			# 检测边界点
			if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
				continue

			if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
				seedMark[tmpX, tmpY] = 255
				seeds.append((tmpX, tmpY))
	return seedMark


if __name__ == '__main__':
	test_hock()

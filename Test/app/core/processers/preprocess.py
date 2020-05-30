# -*- coding: utf-8 -*-
import os
import random

import cv2
import numpy as np

# 发现
from app.config import IMG_HEIGHT, IMG_WIDTH, SUPPORTREFROI_DIR
from app.core.beans.models import SupportRefRoi
from app.core.processers import SmallWords
from app.core.support.shapedetect import ShapeDetector
import ctypes
from ctypes import cdll, c_uint, c_void_p, c_int, c_float, c_char_p, POINTER, byref, Structure, cast, c_uint8


class AbstractDetector(metaclass=SmallWords):
	'''
	AbstractDetector
	：used freequently
	'''
	# OPENCV_SUPPLYDLL = cdll.LoadLibrary(
	# 	"C:/NTY_IMG_PROCESS/dll/libOPENCV_SUPPLY.dll")

	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	# 图像锐化操作
	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst

	# 对灰度图像做数据插值运算
	def interpolation_binary_data(self, binary_image):
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	def enhanceimg(self):
		rows, cols = self.shape
		self.hsv[:, 0:int(0.5 * cols), 2] += 3
		self.hsv[:, int(0.5 * cols + 1):cols, 2] += 10
		self.hsv = self.hsv

	def color_similar_ratio(self, image1, image2):
		'''两张图片的相似度'''
		if image1 is None or image2 is None:
			return 0
		img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		# cv2.imshow("hist1",hist1)
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
		return degree

	def red_contours(self, img, middle_start=180, middle_end=500):
		'''返回红色轮廓'''
		red_low, red_high = [120, 50, 50], [180, 255, 255]
		red_min, red_max = np.array(red_low), np.array(red_high)
		# 去除颜色范围外的其余颜色
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		red_mask = cv2.inRange(hsv, red_min, red_max)
		ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
		middle_open_mask = np.zeros_like(red_binary)
		middle_open_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255
		red_binary = cv2.bitwise_and(red_binary, red_binary, mask=middle_open_mask)
		red_binary = cv2.medianBlur(red_binary, 3)
		red_contours, _hierarchy = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return red_binary, red_contours

	def yellow_contours(self, img):
		'''
		返回黄色轮廓
		:return:
		'''
		yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]

		yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)
		# 去除颜色范围外的其余颜色

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

		yellow_ret, yellow_binary = cv2.threshold(yellow_mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		yellow_binary = cv2.medianBlur(yellow_binary, 3)

		yellow_contours, _hierarchy = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return yellow_binary, yellow_contours

	def green_contours(self, img, middle_start=150, middle_end=500):
		'''
		返回黄色轮廓
		:return:
		'''
		rows, cols, channels = img.shape
		# 如果尺寸已经调整，就无须调整
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		green_low, green_high = [35, 43, 46], [77, 255, 255]
		# green_low, green_high = [17, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		green_mask = cv2.inRange(hsv, green_min, green_max)
		green_ret, binarry = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		binarry = cv2.filter2D(binarry, -1, disc)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		middle_mask=np.zeros_like(gray)
		middle_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255

		foreground = cv2.bitwise_and(binarry, binarry, mask=middle_mask)

		# cv2.imshow("green_binary", foreground)
		foreground = cv2.medianBlur(foreground, 3)
		green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return foreground, green_contours

	# def modify_x(self, img):
	# 	'''
	# 	行车中有两条平行的线，这两个平行的线的x轴可以用来更好的标注袋子的X轴坐标
	# 	:return:
	# 	'''
	#
	# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#
	# 	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	#
	# 	cv2.imshow("canny", edges)
	#
	# 	minLineLength = 200
	# 	maxLineGap = 15
	# 	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
	#
	# 	targetimg = img.copy()
	#
	# 	for x1, y1, x2, y2 in lines[0]:
	# 		cv2.line(targetimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
	#
	# 	cv2.imshow('targetimg', targetimg)
	# 	cv2.waitKey(0)



	@classmethod
	def error_causedby_angel_height(cls, target=None, width_start=130, width_end=160):
		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 100 <= cv2.contourArea(c) < 1600
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 3 < rect_w <= 40 and 3 < rect_h <= 40

		supportref_roi = [SupportRefRoi(img=cv2.imread(os.path.join(SUPPORTREFROI_DIR, roi_img)), id=index)
		                  for
		                  index, roi_img in
		                  enumerate(os.listdir(SUPPORTREFROI_DIR)) if roi_img.find('support') != -1]
		roi_template = random.choice(supportref_roi)

		target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		cols, rows, channels = target.shape
		print(rows, cols, channels)

		one_mask = np.zeros_like(gray)
		one_mask[0:IMG_HEIGHT, width_start:width_end] = 255

		# one_mask[0:IMG_HEIGHT, 160:400] = 255

		# cv2.imshow("one_mask", one_mask)

		must_unique_window = {}
		img_roi_hsvt = cv2.cvtColor(roi_template.roi, cv2.COLOR_BGR2HSV)
		# cv2.imshow("roihist",img_roi_hsvt)
		img_roi_hsvt = img_roi_hsvt
		roihist = cv2.calcHist([img_roi_hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

		cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
		bk = cv2.calcBackProject([target_hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		bk = cv2.filter2D(bk, -1, disc)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

		bk = cv2.bitwise_and(bk, bk, mask=one_mask)

		bk = cv2.dilate(bk, kernel)

		ret, thresh = cv2.threshold(bk, 0, 255, cv2.THRESH_BINARY)

		# cv2.imshow("support", thresh)

		# thresh=cv2.fastNlMeansDenoisingMulti(thresh,2,5,None,4,7,35)

		# 使用merge变成通道图像
		# thresh = cv2.merge((thresh, thresh, thresh))

		# thresh = cv2.medianBlur(thresh, 3)
		# thresh=cv2.bilateralFilter(thresh,d=0,sigmaColor=90,sigmaSpace=7)
		# if isinstance(roi_template,BagRoi):
		# 	cv2.imshow("bag", thresh)

		contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# print("{} contours size {}".format(roi_template.label, len(contours)))
		# cv2.drawContours(target, contours, -1, (0, 255, 255), 3)

		# Z轴无论再怎么变化，灯的面积也大于90
		if contours is not None and len(contours) > 0:
			contours = list(filter(lambda c: warp_filter(c), contours))

		cv2.drawContours(target, contours, -1, (0, 255, 0), 3)

		center_x = center_y = 0
		usefulvaluse = []
		for c in contours:
			M = cv2.moments(c)
			try:
				center_x = int(M["m10"] / M["m00"])
				center_y = int(M["m01"] / M["m00"])
			except:
				continue
			else:
				usefulvaluse.append(center_x)

		distance = np.min(usefulvaluse)
		print("x distance is {}".format(distance))

		return distance if len(usefulvaluse) > 0 else 0

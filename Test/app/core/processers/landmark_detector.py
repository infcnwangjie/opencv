# -*- coding: utf-8 -*-
from itertools import chain

import cv2
import numpy as np

# 发现
from app.core.location.models import LandMark
from app.core.processers.preprocess import Preprocess
from app.core.support.shapedetect import ShapeDetector


class LandMarkDetector(Preprocess):
	'''
	预处理操作都在这里
	'''

	FEATURE_DETECT = 1
	HSV_RANGE = 2

	def __init__(self, img):
		super().__init__(img)
		self.landmarks = []

	def detect(self, method: int):
		if method == self.FEATURE_DETECT:
			pass
		else:
			pass

	def similarity(self, image1, image2):
		if image1 is None or image2 is None:
			return 0
		img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
		return degree

	def slide(self):
		img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		for row in range(0, rows):
			for col in range(502, 612):
				# print("-" * 1000)
				yield (col, row, img[row:row + 80, col:col + 80])

	# for col in range(2619, 2743):
	# 	print("-" * 1000)
	# 	yield (col, row, img[row:row + 80, col:col + 80])

	# 获取已处理过的二值化图像
	@property
	def processedlandmarkimg(self):
		self.enhanceimg()
		red_binary, red_contours = self.red_contours()
		green_binary, green_contours = self.green_contours()
		yellow_binary, yellow_contours = self.yellow_contours()

		if red_contours is None and green_contours is None and yellow_contours is None:
			print("none")
			return

		boxindex = 0
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		land_mark_contours = []
		for countour in chain(green_contours, red_contours, yellow_contours):
			x, y, w, h = cv2.boundingRect(countour)
			if w < 20 or h < 20 or h > 300 or w > 300:
				continue
			if not 400 < cv2.contourArea(countour) < 80000:
				continue
			cent_x, cent_y = x + round(w * 0.5), y + round(h * 0.5)
			if 0.18 * cols < cent_x < 0.23 * cols or 0.85 * cols < cent_x < 0.891 * cols:  # 真实
				land_mark_contours.append(countour)

				box = LandMark(countour, red_binary, id=boxindex, ori_img=self.img)
				box.modify_box_content(no_num=True)
				boxindex += 1
				self.landmarks.append(box)
				cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0] - 200, box.boxcenterpoint[1] + 60),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		# 用黄色画轮廓
		cv2.drawContours(self.img, land_mark_contours, -1, (255, 255, 0), 5)

		return self.landmarks

# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.location.models import Laster
from app.core.processers.preprocess import Preprocess
from app.core.support.shapedetect import ShapeDetector


class LasterDetector(Preprocess):
	'''
	预处理操作都在这里
	'''

	def __init__(self, img):
		super().__init__(img)
		self.laster = None

	@property
	def processed_laster(self):
		def distance_middle(contour):
			rows, cols = self.shape
			x, y, w, h = cv2.boundingRect(contour)
			print(x, y, w, h, abs(cols / 2 - x))
			return abs(cols / 2 - x)

		def filter_laster_contour(c):
			rows, cols = self.shape
			x, y, w, h = cv2.boundingRect(c)
			center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			if w < 10 or h < 15:
				return False
			if  0.2 * cols < center_x < 0.7 * cols:
				return True
			else:
				return False

		binary, contours = self.green_contours()
		goodcontours = list(filter(filter_laster_contour, contours))

		#
		# process = Preprocess(self.img)
		# laster_binary, contours = process.processed_laster
		# cv2.imshow("binary1", laster_binary)
		if goodcontours is None:
			return None

		goodcontours = sorted(goodcontours,
		                      key=lambda c: distance_middle(c), reverse=False)

		cv2.drawContours(self.img, goodcontours, -1, (255, 0, 0), 3)  # 找到唯一的轮廓就退出即可

		laster = Laster(goodcontours[0], binary, id=0)
		laster.modify_box_content()
		self.laster = laster

		return self.laster

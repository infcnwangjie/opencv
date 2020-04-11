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
			if w < 20 or h < 20:
				return False
			if 0.32 * cols < x + 0.5 * w < 0.72 * cols:
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

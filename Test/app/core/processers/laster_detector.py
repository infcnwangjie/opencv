# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.beans.models import Laster
from app.core.processers.preprocess import Preprocess
from app.core.support.shapedetect import ShapeDetector


class LasterDetector(Preprocess):
	'''
	预处理操作都在这里
	'''

	def __init__(self, img):
		super().__init__(img)
		self.laster = None

	def location_laster(self):
		def distance_middle(contour):
			rows, cols = self.shape
			x, y, w, h = cv2.boundingRect(contour)
			print(x, y, w, h, abs(cols / 2 - x))
			return abs(cols / 2 - x)

		def filter_laster_contour(c):
			# print(cols)
			x, y, w, h = cv2.boundingRect(c)
			center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))

			if w < 10 or h < 10 or w > 100 or h > 100:
				return False
			if 160 < center_x < 500:
				# 这里的坐标已经经过标定，数据会比较准确
				return True
			else:
				return False

		binary, contours = self.green_contours()
		goodcontours = list(filter(filter_laster_contour, contours))

		if goodcontours is None:
			return None

		cv2.drawContours(self.img, goodcontours, -1, (255, 0, 0), 3)

		laster = Laster(goodcontours[0], binary, id=0)
		laster.modify_box_content()
		self.laster = laster

		print(self.laster.x, self.laster.y)
		return self.laster

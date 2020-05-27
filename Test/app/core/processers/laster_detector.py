# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.beans.models import Laster
from app.core.processers.preprocess import AbstractDetector
from app.core.support.shapedetect import ShapeDetector


class LasterDetector(AbstractDetector):
	'''
	预处理操作都在这里
	'''

	def __init__(self):
		super().__init__()
		self.laster = None

	def location_laster(self, img_show, img_copy, middle_start=200, middle_end=500):

		def __filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			if w > 5 or h > 5:
				return False
			else:
				return True

		foregroud, contours = self.green_contours(img_copy, middle_start, middle_end)
		contours = list(filter(__filter_laster_contour, contours))

		if contours is None or len(contours) > 1 or len(contours)==0:
			return None, foregroud


		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)

		try:
			self.laster = Laster(contours[0], foregroud, id=0)
			self.laster.modify_box_content()
		except Exception as e:
			print("laster contour is miss")

		return self.laster, foregroud

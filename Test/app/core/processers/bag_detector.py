# -*- coding: utf-8 -*-
import os
from functools import reduce

import cv2
import numpy as np

# 发现
from app.config import BAGROI_DIR, IMG_HEIGHT, IMG_WIDTH
from app.core.processers.preprocess import AbstractDetector
from app.core.beans.models import Bag, LandMarkRoi, BagRoi


def add_picture(img1, img2):
	'''图片叠加'''
	return cv2.add(img1, img2)


class BagDetector(AbstractDetector):
	def __init__(self, img=None):
		super().__init__()
		self.bags = []

	def bagroi_templates(self):
		landmark_rois = [BagRoi(img=cv2.imread(os.path.join(BAGROI_DIR, roi_img)), id=index)
		                 for
		                 index, roi_img in
		                 enumerate(os.listdir(BAGROI_DIR)) if roi_img.find('bag') != -1]
		return landmark_rois

	def findbags(self, img_copy=None, roi_template=None, middle_start=300, middle_end=500):
		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 100 <= cv2.contourArea(c) < 3000
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 8 < rect_w <= 60 and 4 < rect_h <= 60

		global rows, cols, step
		# target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

		gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
		cols, rows, channels = img_copy.shape
		# print(rows,cols,channels)
		foreground, contours = self.red_contours(img_copy, middle_start, middle_end)
		#
		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.filter2D(foreground, -1, disc)
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.dilate(foreground, kernel)

		ret, foreground = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Z轴无论再怎么变化，灯的面积也大于90
		if contours is not None and len(contours) > 0:
			contours = list(filter(lambda c: warp_filter(c), contours))

		return contours, foreground

	def location_bags(self, img_final_show, img_copy, success_location=True, middle_start=150, middle_end=500):
		'''
	     cm:def location_bags(self,target,success_location=True,middle_start=150,middle_end=500):
		'''

		forgounds = []
		for bag_template in self.bagroi_templates():
			# print(target)
			contours, foreground = self.findbags(img_copy, bag_template, middle_start, middle_end)
			# 存储前景图
			forgounds.append(foreground)
			if contours is None or len(contours) == 0:
				continue
			else:
				# contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
				for increased_id, c in enumerate(contours):
					box = Bag(c, img=None, id=increased_id)
					box.modify_box_content(no_num=True)
					if success_location:
						cv2.putText(img_final_show, box.box_content,
						            (box.boxcenterpoint[0], box.boxcenterpoint[1] + 10),
						            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
					self.bags.append(box)

		cv2.drawContours(img_final_show, contours, -1, (0, 255, 255), 3)

		bag_forground = reduce(add_picture, forgounds)
		return self.bags, bag_forground

# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.processers.preprocess import Preprocess
from app.core.beans.models import Bag


class BagDetector(Preprocess):
	def __init__(self, img):
		super().__init__(img)
		self.bags = []


	def location_bag(self):
		bag_binary,contours=self.red_contours()

		if contours is None or len(contours) == 0:
			return
		# 大小适中的轮廓，过小的轮廓被去除了
		moderatesize_countours = []
		boxindex = 0
		rows, cols = bag_binary.shape
		for countour in contours:
			countour_rect = cv2.boundingRect(countour)
			rect_x, rect_y, rect_w, rect_h = countour_rect
			center_x, center_y = (rect_x + round(rect_w * 0.5), rect_y + round(rect_h * 0.5))
			# cv2.contourArea(countour) > 500 and rect_h < 300 and
			if  200 < center_x < 500:
				moderatesize_countours.append(countour)
				boxindex += 1
				box = Bag(countour, bag_binary, id=boxindex)

				box.modify_box_content(no_num=True)
				cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0] + 50, box.boxcenterpoint[1] + 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
				self.bags.append(box)

		cv2.drawContours(self.img, moderatesize_countours, -1, (0, 255, 255), 3)
		print("i have  try my best")
		return self.bags

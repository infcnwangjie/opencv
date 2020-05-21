# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

# 发现
from app.config import BAGROI_DIR, IMG_HEIGHT, IMG_WIDTH
from app.core.processers.preprocess import AbstractDetector
from app.core.beans.models import Bag, LandMarkRoi, BagRoi


class BagDetector(AbstractDetector):
	def __init__(self, img=None):
		super().__init__()
		self.bags = []

	def bagroi_templates(self):
		landmark_rois = [BagRoi(img=cv2.imread(os.path.join(BAGROI_DIR, roi_img)), id=index)
		                 for
		                 index, roi_img in
		                 enumerate(os.listdir(BAGROI_DIR)) if roi_img.find('bag') !=-1]
		return landmark_rois


	def findbags(self, target=None, roi_template=None,middle_start=100,middle_end=400):
		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 30 <= cv2.contourArea(c) < 3000
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 6 < rect_w <= 60 and 6 < rect_h <= 60

		global rows, cols, step
		# target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		cols,rows,channels=target.shape
		# print(rows,cols,channels)
		foreground, contours = self.red_contours(target,middle_start,middle_end)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.dilate(foreground, kernel)

		ret, thresh = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)

		contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Z轴无论再怎么变化，灯的面积也大于90
		if contours is not None and len(contours) > 0:
			contours = list(filter(lambda c: warp_filter(c), contours))

		return contours

	def location_bags(self,target,success_location=True,middle_start=150,middle_end=500):
		'''
		location_bags
		:return [Bag]
		袋子检测不能完全依赖于透视变换，地标检测也不能完全的作为袋子的参照物;
		从摄像头垂直向下的时候，袋子多出来的X轴误差，就是摄像头中的地标并不是完全垂直于地面造成的
		'''
		moderatesize_countours = []
		# x_error=AbstractDetector.error_causedby_angel_height(target)
		x_error=0
		for bag_template in self.bagroi_templates():
			# print(target)
			if success_location:
				contours = self.findbags(target, bag_template,middle_start,middle_end)
			else:
				contours = self.findbags(target, bag_template, 380, 500)
			if contours is None or len(contours) == 0:
				continue
			else:
				# contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
				moderatesize_countours.extend(contours)
				for increased_id, c in enumerate(contours):
					box = Bag(c, img=None, id=increased_id)
					if x_error is  not None:
						box.x=box.x-int(x_error)
					box.modify_box_content(no_num=True)
					if success_location:
						cv2.putText(target, box.box_content, (box.boxcenterpoint[0], box.boxcenterpoint[1] + 10),
						            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
					self.bags.append(box)

		cv2.drawContours(target, moderatesize_countours, -1, (0, 255, 255), 3)
		# print("i have  try my best")
		# self.modify_x(self.img)

		return self.bags

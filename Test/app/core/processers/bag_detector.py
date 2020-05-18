# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

# 发现
from app.config import BAGROI_DIR
from app.core.processers.preprocess import AbstractDetector
from app.core.beans.models import Bag, LandMarkRoi, BagRoi


class BagDetector(AbstractDetector):
	def __init__(self, img):
		super().__init__(img)
		self.bags = []

	def bagroi_templates(self):
		landmark_rois = [BagRoi(img=cv2.imread(os.path.join(BAGROI_DIR, roi_img)), id=index)
		                 for
		                 index, roi_img in
		                 enumerate(os.listdir(BAGROI_DIR))]
		return landmark_rois

	def location_bag(self):
		'''
		location_bag
		:return [Bag]
		袋子检测不能完全依赖于透视变换，地标检测也不能完全的作为袋子的参照物;
		从摄像头垂直向下的时候，袋子多出来的X轴误差，就是摄像头中的地标并不是完全垂直于地面造成的
		'''
		moderatesize_countours = []
		for bag_template in self.bagroi_templates():
			contours = self.find_it(self.img, bag_template)
			if contours is None or len(contours) == 0:
				continue
			else:
				# contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
				moderatesize_countours.extend(contours)
				for increased_id, c in enumerate(contours):
					box = Bag(c, img=None, id=increased_id)
					# TODO 记录袋子的位置
					box.modify_box_content(no_num=True)
					cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0], box.boxcenterpoint[1] + 10),
					            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
					self.bags.append(box)

		cv2.drawContours(self.img, moderatesize_countours, -1, (0, 255, 255), 3)
		# print("i have  try my best")
		# self.modify_x(self.img)

		return self.bags

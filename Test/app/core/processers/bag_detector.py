# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np

# 发现
from app.config import BAGROI_DIR, IMG_HEIGHT, IMG_WIDTH
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
		                 enumerate(os.listdir(BAGROI_DIR)) if roi_img.find('bag') !=-1]
		return landmark_rois

	def findit(self, target=None, roi_template=None):
		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 80 <= cv2.contourArea(c) < 300
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 3 < rect_w <= 30 and 3 < rect_h <= 30

		global rows, cols, step
		target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

		left_open_mask = np.zeros_like(gray)
		left_open_mask[0:IMG_HEIGHT, 0:300] = 255

		right_open_mask = np.zeros_like(gray)
		right_open_mask[0:IMG_HEIGHT, 700:IMG_WIDTH] = 255

		middle_open_mask = np.zeros_like(gray)
		middle_open_mask[0:IMG_HEIGHT, 150:500] = 255

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
		if hasattr(roi_template, 'label'):
			if roi_template.label.find("L") > 0:
				bk = cv2.bitwise_and(bk, bk, mask=left_open_mask)
			if roi_template.label.find("R") > 0:
				bk = cv2.bitwise_and(bk, bk, mask=right_open_mask)
		else:
			bk = cv2.bitwise_and(bk, bk, mask=middle_open_mask)
		bk = cv2.dilate(bk, kernel)

		ret, thresh = cv2.threshold(bk, 50, 255, cv2.THRESH_BINARY)

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
		# if contours is not None and len(contours) > 0:
		# 	contours = list(filter(lambda c: warp_filter(c), contours))

		return contours

	def location_bags(self):
		'''
		location_bags
		:return [Bag]
		袋子检测不能完全依赖于透视变换，地标检测也不能完全的作为袋子的参照物;
		从摄像头垂直向下的时候，袋子多出来的X轴误差，就是摄像头中的地标并不是完全垂直于地面造成的
		'''
		moderatesize_countours = []
		for bag_template in self.bagroi_templates():
			contours = self.findit(self.img, bag_template)
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

from functools import partial

import cv2

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.beans.models import LandMarkRoi, BagRoi
from app.log.logtool import logger
import numpy as np


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
	middle_open_mask[0:IMG_HEIGHT, 120:500] = 255

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
	# if roi_template.label=="NO3_R":
	# 	cv2.imshow("bk", thresh)
	ret, thresh = cv2.threshold(bk, 100, 255, cv2.THRESH_BINARY)
	# thresh=cv2.fastNlMeansDenoisingMulti(thresh,2,5,None,4,7,35)

	# 使用merge变成通道图像
	# thresh = cv2.merge((thresh, thresh, thresh))

	# thresh = cv2.medianBlur(thresh, 3)
	# thresh=cv2.bilateralFilter(thresh,d=0,sigmaColor=90,sigmaSpace=7)
	if isinstance(roi_template,BagRoi):
		cv2.imshow("bag", bk)

	contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# print("{} contours size {}".format(roi_template.label, len(contours)))
	# cv2.drawContours(target, contours, -1, (0, 255, 255), 3)

	# Z轴无论再怎么变化，灯的面积也大于90
	# if contours is not None and len(contours) > 0:
	# 	contours = list(filter(lambda c: warp_filter(c), contours))

	return contours


class SmallWords(type):
	def __new__(cls, name, bases, attrs):
		if attrs is None:
			attrs = {}
		attrs['logger'] = logger
		attrs['find_it'] = findit
		cls.instance = None

		return super().__new__(cls, name, bases, attrs)

	def __call__(self, *args, **kwargs):
		if self.instance is not None:
			return self.instance
		else:
			return super().__call__(*args, **kwargs)

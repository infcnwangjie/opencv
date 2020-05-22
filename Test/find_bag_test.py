# -*- coding: utf-8 -*-
# from gevent import monkey;
import os
import pickle
import random
from collections import defaultdict
from functools import cmp_to_key
from operator import itemgetter

import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, ROIS_DIR, LEFT_MARK_FROM, LEFT_MARK_TO, RIGHT_MARK_FROM, RIGHT_MARK_TO, \
	PROGRAM_DATA_DIR, SUPPORTREFROI_DIR
import cv2
import time
import gevent
import profile
from app.core.beans.models import LandMarkRoi, NearLandMark, TargetRect
from app.core.exceptions.allexception import NotFoundLandMarkException
from app.core.processers.bag_detector import BagDetector
from app.core.processers.preprocess import AbstractDetector
from app.log.logtool import mylog_error
import re

cv2.useOptimized()

rows, cols = IMG_HEIGHT, IMG_WIDTH


def compute_bag_location(target=None, start=0, end=400):
	def warp_filter(c):
		'''内部过滤轮廓'''
		isbig = 70 <= cv2.contourArea(c) < 600
		rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
		return isbig and 5 < rect_w <= 100 and 5 < rect_h <= 100
		return isbig

	target = cv2.resize(target, (IMG_HEIGHT, IMG_WIDTH))

	detector = AbstractDetector()
	foregroud, contours = detector.red_contours(target)
	# cv2.imshow("red_binary", foregroud)

	mask = np.zeros_like(foregroud)
	mask[0:IMG_HEIGHT, start:end] = 255

	bag_foregroud = cv2.bitwise_and(foregroud, foregroud, mask=mask)
	cv2.imshow("bag_foregroud", bag_foregroud)

	bag_contours, _hierarchy = cv2.findContours(bag_foregroud, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if bag_contours is not None and len(bag_contours) > 0:
		bag_contours = list(filter(lambda c: warp_filter(c), bag_contours))

	cv2.drawContours(target, bag_contours, -1, (0, 255, 0), 3)

	return bag_contours


# cv2.namedWindow("dest")
# cv2.imshow("dest", target)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def landmark_location(target=None, left_start=130, left_end=200, right_start=550, right_end=600,y_start=385):
	def warp_filter(c):
		'''内部过滤轮廓'''
		isbig = 70 <= cv2.contourArea(c) < 600
		rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
		return isbig and 5 < rect_w <= 100 and 5 < rect_h <= 100
		return isbig

	target = cv2.resize(target, (IMG_HEIGHT, IMG_WIDTH))

	detector = AbstractDetector()
	foregroud, contours = detector.yellow_contours(target)
	cv2.imshow("yellow_binary", foregroud)

	landmark_mask = np.zeros_like(foregroud)
	landmark_mask[y_start:IMG_HEIGHT, left_start:left_end] = 255
	landmark_mask[y_start:IMG_HEIGHT, right_start:right_end] = 255

	landmark_foregroud = cv2.bitwise_and(foregroud, foregroud, mask=landmark_mask)
	cv2.imshow("landmark_foregroud", landmark_foregroud)

	landmark_contours, _hierarchy = cv2.findContours(landmark_foregroud, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if landmark_contours is not None and len(landmark_contours) > 0:
		landmark_contours = list(filter(lambda c: warp_filter(c), landmark_contours))

	cv2.drawContours(target, landmark_contours, -1, (0, 255, 0), 3)

	# position_dict={}
	# positiondict[compensate_label] = [miss_x, miss_y]
	points = []
	for c in landmark_contours:
		rect = cv2.boundingRect(c)
		x, y, w, h = rect
		points.append((x, y))

	left_points = [(x, y) for x, y in points if x < 200]
	left_points = sorted(left_points, key=itemgetter(1), reverse=False)

	right_points = [(x, y) for x, y in points if x > 500]
	right_points = sorted(right_points, key=itemgetter(1), reverse=False)

	pts1 = np.float32([left_points[0], left_points[1], right_points[0], right_points[1]])
	pts2 = np.float32([[0, 0], [0, 200],
	                   [574, 0], [574, 200]])

	# 生成透视变换矩阵；进行透视变换
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(target, M, (700, 900))

	bagcontours = compute_bag_location(dst)

	for bag in bagcontours:
		rect = cv2.boundingRect(bag)
		x, y, w, h = rect
		cv2.rectangle(dst, (x, y), (x + w, y + h), color=(0, 255, 255),
		              thickness=2)
		cv2.putText(dst,
		           "({},{})".format(x, y),
		           (x, y + 20),
		           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		# cv2.putText(dest,
		#             "{}".format(landmark_roi.label),
		#             (col, row + 60),
		#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

	return dst


if __name__ == '__main__':
	target = cv2.imread("c:/work/nty/hangche/2020-05-20-10-26-39test.bmp")
	target = cv2.resize(target, (IMG_HEIGHT, IMG_WIDTH))
	# compute_bag_location(target)
	dest = landmark_location(target)
	compute_bag_location(dest)
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

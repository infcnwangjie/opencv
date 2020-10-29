# -*- coding: utf-8 -*-
import math
import pickle
import time

import cv2
import numpy as np
import re
import ctypes
from ctypes import cdll
from collections import defaultdict
from functools import cmp_to_key, reduce, partial
import os

from app.core.autowork import f
from app.core.autowork.processhandle import *
from app.core.beans.models import *
from app.core.plc.plchandle import PlcHandle
from app.core.support.shapedetect import ShapeDetector
from app.core.video.mvs.MvCameraSuppl_class import MvSuply
from app.log.logtool import mylog_error, mylog_debug, logger

from app.config import *
from app.core.exceptions.allexception import NotFoundLandMarkException

from app.log.logtool import mylog_error

cv2.useOptimized()
rows, cols = IMG_HEIGHT, IMG_WIDTH
WITH_TRANSPORT = True

landmark_pattern = re.compile("""NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})""")



def shrink_coach_area(perspect_img):
	def inner_filter(c):
		x, y, w, h = cv2.boundingRect(c)
		if w < 100 or h < 100:
			return False
		area=cv2.contourArea(c)
		# print("area:{}".format(area))
		if area < 60000:
			return False
		return True

	gray = cv2.cvtColor(perspect_img, cv2.COLOR_BGR2GRAY)

	ret, white_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	foreground = cv2.dilate(white_binary, kernel)

	middle_open_mask = np.zeros_like(gray)
	middle_open_mask[0:rows, middle_start_withlandmark:middle_end_withlandmark] = 255

	foreground = cv2.bitwise_and(foreground, foreground, mask=middle_open_mask)

	contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(list(filter(lambda c: inner_filter(c), contours)), key=lambda c: cv2.contourArea(c),
	                  reverse=True)
	contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

	if len(contours) == 0: raise Exception("not find big white area")
	black_empty = np.zeros_like(gray)
	try:
		cv2.fillPoly(black_empty, [contours[0]], (255, 255, 255))
	except Exception as e:
		cv2.putText(perspect_img, "ERROR:ON_CAR is True", (300, 700),
		            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

	return black_empty


# ------------------------------------------------
# 名称：BaseDetector
# 功能：目标检测基类，提供一些公共的方法
# 状态：在用
# 作者：王杰  注释后补
# ------------------------------------------------
class BaseDetector(object):
	orb_detector = None  #
	instance = None

	def __new__(cls, *args, **kwargs):
		if cls.instance == None:
			cls.instance = super().__new__(cls)
		return cls.instance

	# ------------------------------------------------
	# 名称：logger
	# 功能：封装日志功能
	# 状态：在用
	# 参数： [msg]   ---日志消息
	#        [lever] ---日志级别
	# 返回：None ---
	# 作者：王杰  2020-5-xx
	# ------------------------------------------------
	def logger(self, msg: str, lever='info'):
		from app.log.logtool import logger
		logger(msg, lever)


	# 获取图像行、列总数
	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols


	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst


	def interpolation_binary_data(self, binary_image):
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	def color_similar_ratio(self, image1, image2):
		if image1 is None or image2 is None:
			return 0
		try:
			degree = float(MvSuply.SAME_RATE(image1, image2))
		except Exception as e:
			degree = 0
		return degree

	# ------------------------------------------------
	# 名称：red_contours
	# 状态：在用
	# 参数： [array]
	#        [int]
	#        [int]
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def red_contours(self, img, middle_start=200, middle_end=500):
		red_low, red_high = [156, 43, 46], [180, 255, 255]
		red_min, red_max = np.array(red_low), np.array(red_high)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		red_mask = cv2.inRange(hsv, red_min, red_max)
		ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
		middle_open_mask = np.zeros_like(red_binary)
		middle_open_mask[:, middle_start:middle_end] = 255
		red_binary = cv2.bitwise_and(red_binary, red_binary, mask=middle_open_mask)
		red_binary = cv2.medianBlur(red_binary, 3)
		red_contours, _hierarchy = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return red_binary, red_contours

	# ------------------------------------------------
	# 名称：red_contours
	# 状态：在用
	# 参数： [array]   ---输入图像
	# 要求： img是RGB格式图片
	# 返回： 数组     ---轮廓数组
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def yellow_contours(self, img, middle_start=200, middle_end=500):
		yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]

		yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

		yellow_ret, yellow_binary = cv2.threshold(yellow_mask, 100, 255, cv2.THRESH_BINARY)

		middle_open_mask = np.zeros_like(yellow_binary)
		middle_open_mask[:, middle_start:middle_end] = 255
		yellow_binary = cv2.bitwise_and(yellow_binary, yellow_binary, mask=middle_open_mask)

		yellow_contours, _hierarchy = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return yellow_binary, yellow_contours


	def green_contours(self, img, middle_start=100, middle_end=450):
		rows, cols, channels = img.shape

		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		green_mask = cv2.inRange(hsv, green_min, green_max)

		green_ret, foreground = cv2.threshold(green_mask, 100, 255, cv2.THRESH_BINARY)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)

		green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return foreground, green_contours

	def is_single_color_obj(self, c, hsv_img):
		match_times = 0

		if np.any(hsv_img) == False: return False

		x, y, w, h = cv2.boundingRect(c)

		if w < 7 or h < 7: return False
		for key, (startarray, endarray) in COLOR_RANGE.items():
			min_, max_ = np.array(startarray), np.array(endarray)
			range_ = cv2.inRange(hsv_img[y:y + h - 1, x:x + w - 1, :], min_, max_)
			range_ret, foreground = cv2.threshold(range_, 0, 255, cv2.THRESH_BINARY)
			disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			foreground = cv2.filter2D(foreground, -1, disc)
			contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if len(contours) > 1: match_times += 1

		return not match_times > 1

	def point_in_contour(self, x, y, c):
		c_x, c_y, c_w, c_h = cv2.boundingRect(c)

		top1_x, top1_y, top2_x, top2_y = c_x, c_y, c_x + c_w, c_y
		top3_x, top3_y, top4_x, top4_y = c_x, c_y + c_h, c_x + c_w, c_y + c_h

		if top1_x < x < top2_x and top1_y < y < top3_y:
			return True
		else:
			return False

	def point_in_rect(self, x, y, c_x, c_y, c_w, c_h):
		top1_x, top1_y, top2_x, top2_y = c_x, c_y, c_x + c_w, c_y
		top3_x, top3_y, top4_x, top4_y = c_x, c_y + c_h, c_x + c_w, c_y + c_h

		if top1_x < x < top2_x and top1_y < y < top3_y:
			return True
		else:
			return False


	def color_match_result(self, target, color_code, c):
		hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
		colorinfo = COLOR_RANGE[color_code]

		rect = cv2.boundingRect(c)
		x, y, w, h = rect

		area = cv2.contourArea(c)

		color_code_len = len(color_code.split('_'))

		test_roi = hsv[y:y + 100, x:x + 100]

		# test_roi = hsv[y - 50:y + 50, x - 50:x + 50]
		if color_code_len == 1:
			color_low, color_high = colorinfo
			color_match = self.contour_in_range(area, color_low, color_high, test_roi)
			return color_match
		else:
			color1_low, color1_high = colorinfo[0]

			color1_match = self.contour_in_range(area, color1_low, color1_high, test_roi)

			color2_low, color2_high = colorinfo[1]
			color2_match = self.contour_in_range(area, color2_low, color2_high, test_roi)

			return color1_match and color2_match


	def contour_in_range(self, area, color1_low, color1_high, test_roi):
		if area == 0:
			return False
		color1_min, color1_max = np.array(color1_low), np.array(color1_high)
		color1_mask = cv2.inRange(test_roi, color1_min, color1_max)
		ret, color1_foreground = cv2.threshold(color1_mask, 0, 255, cv2.THRESH_BINARY)
		color1_contours, _hierarchy = cv2.findContours(color1_foreground, cv2.RETR_EXTERNAL,
		                                               cv2.CHAIN_APPROX_SIMPLE)
		color1_contours_areadesc = sorted(color1_contours, key=lambda c: cv2.contourArea(c), reverse=True)
		color1_match = False
		if len(color1_contours_areadesc) == 0:
			return False
		test_area = cv2.contourArea(color1_contours_areadesc[0])
		if test_area < 100:
			return False
		if test_area >= 0.2 * area:
			# print(" test_area:{},zhanbi:{}%".format(test_area, test_area / area * 100))
			color1_match = True
		else:
			color1_match = False
		return color1_match

	def calc_center(self, c):
		area = cv2.contourArea(c)
		rect = cv2.boundingRect(c)
		x, y, w, h = rect
		M = cv2.moments(c)
		try:
			center_x = int(M["m10"] / M["m00"])
			center_y = int(M["m01"] / M["m00"])
		except:
			center_x, center_y = x, y
		return center_x, center_y


# 袋子
class BagDetector(BaseDetector):

	def __init__(self, img=None):
		super().__init__()
		self.bags = []
		self.has_init = False
		self.has_update = False
		self.has_stable = False

	def load_or_update_position(self, p_tu=None):
		if self.has_init == False:
			self.f = f()
			self.f.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
			self.f.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
			self.f.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
			self.f.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
			self.has_init = True

		if p_tu is not None:

			x, y = self.get_predict()
			if abs(x - p_tu[0]) < 100 and abs(y - p_tu[1]) < 100:
				self.has_stable = True
			else:
				self.has_stable = False
			pos = np.array([*p_tu], dtype=np.float32)
			mes = np.reshape(pos, (2, 1))
			self.f.correct(mes)
			self.has_update = True

	def get_predict(self):
		guass_position = self.f.predict()
		x, y = int(guass_position[0][0]), int(guass_position[1][0])
		return x, y

	def bagroi_templates(self):
		landmark_rois = [BagRoi(img=cv2.imread(os.path.join(BAGROI_DIR, roi_img)), id=index)
		                 for
		                 index, roi_img in
		                 enumerate(os.listdir(BAGROI_DIR)) if roi_img.find('bag') != -1]
		return landmark_rois

	def findbags(self, img_copy=None):
		hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

		# warp_filter=lambda c: self.is_single_color_obj(c,hsv_img)
		outer_width = outer_height = 10

		def warp_filter(c, hsv_img):
			x, y, w, h = cv2.boundingRect(c)
			target_roi = img_copy[y - outer_width:y + h + outer_height,
			             x - outer_width:x + w + outer_width, :]
			value = MvSuply.SAME_RATE(target_roi, self.bagroi_templates()[0].roi)
			siglecolor = self.is_single_color_obj(c, hsv_img)
			return (siglecolor or value > 0) and middle_start_withlandmark < x < middle_end_withlandmark

		global rows, cols, step

		cols, rows, channels = img_copy.shape
		# print(rows,cols,channels)
		foreground, contours = self.red_contours(img_copy, middle_start_withlandmark, middle_end_withlandmark)

		# ret, foreground = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)

		try:
			foreground_bagarea = shrink_coach_area(img_copy)
			cv2.imshow("foreground_bagarea", foreground_bagarea)
		except Exception as e:
			print(e.__str__())
		else:
			foreground = cv2.bitwise_and(foreground, foreground, mask=foreground_bagarea)
		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# if contours is not None and len(contours) > 0:
		# 	logger("bag contour is {}".format(len(contours)), level='info')

		contours = list(filter(lambda c: warp_filter(c, hsv_img), contours))
		# contours = list(filter(lambda c: warp_filter(c, hsv_img), contours))

		if contours is None or len(contours) == 0:
			foreground, cs = self.yellow_contours(img_copy, middle_start_withlandmark, middle_end_withlandmark)
			if cs is not None and len(cs) > 0: contours = cs

		positions = []
		for c in contours:
			area = cv2.contourArea(c)
			x, y, w, h = cv2.boundingRect(c)

			if x < middle_start_withlandmark or (x + w) > middle_end_withlandmark:
				continue

			area_match = (bag_min_area < area < bag_max_area)
			if area_match == False:
				continue
			# cv2.drawContours(img_copy, [c], -1, (170, 0, 255), 3)
			# rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			cent_x, cent_y = self.calc_center(c)
			positions.append([cent_x, cent_y])

		from app.core.autowork.divide_points import divide_bags
		if len(positions) > 0:
			clusters = divide_bags(np.array(positions))
			bagclusters = [BagCluster(int(c['centroid'][0]), int(c['centroid'][1])) for c in clusters]
		else:
			bagclusters = None
		return bagclusters, contours, foreground

	def location_bags_withlandmark(self, dest, img_copy, success_location=True,
	                               hock=None, plchandle=None
	                               ):
		bagclusters, contours, foreground = self.findbags(img_copy)

		# cv2.drawContours(dest, contours, -1, (170, 0, 255), 3)

		if bagclusters is None:
			return self.bags, foreground

		orderby_x_bagclusters = sorted(bagclusters, key=lambda bagcluster: bagcluster.cent_x,
		                               reverse=False)
		for x_index, orderby_x_c in enumerate(orderby_x_bagclusters):
			orderby_x_c.col_sort_index = x_index + 1

		orderby_y_bagclusters = sorted(orderby_x_bagclusters, key=lambda bagcluster: bagcluster.cent_y,
		                               reverse=False)
		for y_index, orderby_y_c in enumerate(orderby_y_bagclusters):
			orderby_y_c.row_sort_index = y_index + 1

		# self.hock.center_x, self.hock.center_y

		for bag_cluster in orderby_y_bagclusters:
			bag = Bag(centx=bag_cluster.cent_x, centy=bag_cluster.cent_y,
			          id="{}_{}".format(bag_cluster.col_sort_index, bag_cluster.row_sort_index))
			bag.modify_box_content()
			if success_location:

				if hock is not None:
					hock_x, hock_y = hock.center_x, hock.center_y
					if abs(hock_x - bag.cent_x) < 30 and abs(hock_y - bag.cent_y) < 30:
						bag.height = plchandle.get_high()
						bag.modify_box_content()

				# cv2.rectangle(dest, (bag.cent_x-30, bag.cent_y-30),(bag.cent_x, bag.cent_y-20), (	72,209,204), thickness=-1)
				cv2.rectangle(dest, (bag.cent_x - 20, bag.cent_y - 20), (bag.cent_x + 20, bag.cent_y + 20),
				              (30, 144, 255), 3)
				cv2.putText(dest, bag.box_content,
				            (bag.cent_x - 20, bag.cent_y - 25),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

			for existbag in [item for item in self.bags if item.status_map['finish_move'] == False]:
				if abs(existbag.cent_x - bag.cent_x) < 20 and abs(existbag.cent_y - bag.cent_y) < 20:
					existbag.cent_x, existbag.cent_y = bag.cent_x, bag.cent_y
					if bag.cent_x is not None and bag.cent_y is not None:
						existbag.cent_x, existbag.cent_y = bag.cent_x, bag.cent_y
					break
			else:
				self.bags.append(bag)

		return self.bags, foreground

	# ------------------------------------------------
	# 名称：location_bags_withoutlandmark
	# 状态：在用
	# 返回：  [数组]
	#         [bagforeground]
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_bags_withoutlandmark(self, original_img, middle_start=245, middle_end=490):

		bagclusters, contours, foreground = self.findbags(original_img, middle_start, middle_end)

		if bagclusters is None:
			return [], foreground

		orderby_x_bagcluster = sorted(bagclusters, key=lambda bagcluster: bagcluster.cent_x,
		                              reverse=False)
		for x_index, orderby_x_c in enumerate(orderby_x_bagcluster):
			orderby_x_c.col_sort_index = x_index + 1

		orderby_y_bagclusters = sorted(orderby_x_bagcluster, key=lambda bagcluster: bagcluster.cent_y,
		                               reverse=False)
		for y_index, orderby_y_c in enumerate(orderby_y_bagclusters):
			orderby_y_c.row_sort_index = y_index + 1

		bags_withoutlandmark = []
		for bag_cluster in orderby_y_bagclusters:
			bag = Bag(centx=bag_cluster.cent_x, centy=bag_cluster.cent_y,
			          id="{}_{}".format(bag_cluster.col_sort_index, bag_cluster.row_sort_index))
			bag.cent_x = bag_cluster.cent_x
			bag.cent_y = bag_cluster.cent_y
			bag.modify_box_content()

			cv2.putText(original_img, bag.box_content, (bag.cent_x, bag.cent_y + 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

			bags_withoutlandmark.append(bag)

		return bags_withoutlandmark, foreground


# ------------------------------------------------
# 名称：LandMarkDetecotr
# 功能：地标检测算法.
# 状态：在用
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class LandMarkDetecotr(BaseDetector):
	'''
	地标检测算法
	'''
	landmark_match = re.compile(r'''NO([0-9]*)_([A-Z]{1})''')

	def __init__(self):
		self.img_after_modify = None
		self._rois = []
		self.ALL_LANDMARKS_DICT = {}
		self.ALL_POSITIONS = {}

	# ------------------------------------------------
	# 名称： landmarkname_cmp
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def landmarkname_cmp(self, a, b):

		result_a = re.match(self.landmark_match, a[0])

		result_b = re.match(self.landmark_match, b[0])

		if result_a is None and result_b is None:
			return 0
		elif result_a is not None:
			return -1
		elif result_b is not None:
			return 1
		else:
			a_no = int(result_a.group(1))
			b_no = int(result_b.group(1))
			if a_no > b_no:
				return 1
			elif a_no == b_no:
				return 0
			else:
				return -1

	# ------------------------------------------------
	# 名称： corners_levelfour
	# 功能： 检测四个角点
	# ------------------------------------------------
	def corners_levelfour(self, left_top_landmark_name):
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=1)]

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}

		HAS_POINT_NOT_EXIST = False
		for label_item in labels:
			if label_item not in find:
				HAS_POINT_NOT_EXIST = True
				break
		else:
			HAS_POINT_NOT_EXIST = False

		if not HAS_POINT_NOT_EXIST:
			return labels
		else:
			return None

	# ------------------------------------------------
	# 名称： corners_levelsix
	# 功能： 检测四个角点
	# ------------------------------------------------
	def corners_levelsix(self, left_top_landmark_name):

		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=2),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=2)]

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}

		HAS_POINT_NOT_EXIST = False
		for label_item in labels:
			if label_item not in find:
				HAS_POINT_NOT_EXIST = True
				break
		else:
			HAS_POINT_NOT_EXIST = False

		if not HAS_POINT_NOT_EXIST:
			return labels
		else:
			return None

	# ------------------------------------------------
	# 名称： corners_leveleight
	# 功能： 检测四个角点
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def corners_leveleight(self, left_top_landmark_name):
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=3),
		          self.__fetch_neigbbour(left_top_landmark_name, west_step=3)]
		return labels

	# ------------------------------------------------
	# 名称： __fetch_neigbbour
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def __fetch_neigbbour(self, landmark_name, sourth_step: int = 0, west_step: int = 0):
		result = re.match(self.landmark_match, landmark_name)
		if result is None:
			return landmark_name
		current_no = int(result.group(1))
		current_direct = result.group(2)
		# next_direct = "R" if current_direct == "L" else "L"
		direct = "R" if sourth_step == 1 else current_direct

		# no = current_no + west_step if west_step > 0 else current_no
		no = current_no + west_step
		landmark_labelname = "NO{NO}_{D}".format(NO=no, D=direct)
		return landmark_labelname

	# ------------------------------------------------
	# 名称： get_next_no
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
	def get_next_no(self, landmark_name, forward=False):
		result = re.match(self.landmark_match, landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))
		next_no = current_no + 1 if not forward else current_no - 1
		next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
		return next_landmark

	# ------------------------------------------------
	# 名称： get_opposite_landmark
	# 作者：王杰  编写 2020-4-xx
	# ------------------------------------------------
	def get_opposite_landmark(self, landmark_name):
		import re
		result = re.match(self.landmark_match, landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))
		current_d = result.group(2)
		opposite_d = 'R' if current_d == 'L' else 'L'
		next_landmark = "NO{NO}_{D}".format(NO=current_no, D=opposite_d)
		return next_landmark

	# ------------------------------------------------
	# 名称： compute_miss_landmark_position
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
	def compute_miss_landmark_position(self, landmark_name):
		opposite = self.get_opposite_landmark(landmark_name)
		if opposite not in self.ALL_LANDMARKS_DICT:
			raise NotFoundLandMarkException("opposite landmark is not exist")
		y = self.ALL_LANDMARKS_DICT[opposite].row

		rows = int(len(self.rois) / 2)
		x = 0
		index = -rows
		while index < rows:
			if index < 0:
				label = self.get_next_no(landmark_name, forward=True)
			else:
				label = self.get_next_no(landmark_name, forward=False)

			try:
				x = self.ALL_LANDMARKS_DICT[label].col
				break
			except:
				index += 1
				continue

		return x, y

	# ------------------------------------------------
	# 名称： choose_best_cornors
	# 作者：王杰  编写 2020-4-xx  修改 2020-5-xx
	# ------------------------------------------------
	def choose_best_cornors(self):

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}
		landmark_total = sum(find.values())
		if landmark_total < 3:
			return {}, False

		level_four_min_loss, level_six_min_loss, best_four_label_choose, best_six_label_choose = 4, 6, None, None

		best_four_label_choose, best_six_label_choose, loss = self.position_loss(best_four_label_choose,
		                                                                         best_six_label_choose, find,
		                                                                         level_four_min_loss,
		                                                                         level_six_min_loss)
		positiondict = {}
		if best_four_label_choose not in find and best_six_label_choose not in find:
			return {}, False
		elif best_four_label_choose in find and best_six_label_choose in find:
			labels = self.corners_levelfour(best_four_label_choose) if loss[best_four_label_choose]['4'] < \
			                                                           loss[best_six_label_choose][
				                                                           '6'] else self.corners_levelsix(
				best_six_label_choose)
		elif (best_four_label_choose not in find and best_six_label_choose in find) or (
				best_four_label_choose in find and best_six_label_choose not in find):
			labels = self.corners_levelfour(
				best_four_label_choose) if best_four_label_choose in find else self.corners_levelsix(
				best_six_label_choose)

		compensate_label = ""
		for label in labels:
			if label not in self.ALL_LANDMARKS_DICT:
				compensate_label = label
				continue
			else:
				positiondict[label] = [self.ALL_LANDMARKS_DICT[label].col, self.ALL_LANDMARKS_DICT[label].row]

		if compensate_label != "":
			try:
				miss_x, miss_y = self.compute_miss_landmark_position(compensate_label)
			except (NotFoundLandMarkException) as e:
				return {}, False
			else:
				positiondict[compensate_label] = [miss_x, miss_y]

		success = True

		for key, [x, y] in positiondict.items():
			key_result = re.match(self.landmark_match, key)
			key_no = key_result.group(1)
			key_direct = key_result.group(2)
			for key_j, [xj, yj] in positiondict.items():
				keyj_result = re.match(self.landmark_match, key_j)
				keyj_no = keyj_result.group(1)
				keyj_direct = keyj_result.group(2)

				if key == key_j:
					continue
				if (int(key_no) < int(keyj_no) and y > yj) or (
						int(key_no) > int(keyj_no) and y < yj):  # or (key_no==keyj_no and abs(y-yj)>100)
					return positiondict, False

				if key_no == keyj_no and key_direct != keyj_direct and abs(y - yj) > 100:
					return positiondict, False
				if key_j != key and abs(x - xj) < 50 and abs(y - yj) < 50:
					return positiondict, False

				if key_direct == keyj_direct:
					# q = math.sqrt(math.pow(abs(xj - x), 2) + math.pow(abs(yj - y), 2))
					# if q < 100:
					# 	return {}, False

					if abs(int(key_no) - int(keyj_no)) == 1 and abs(yj - y) > 400:
						return {}, False

		position_row_table = defaultdict(list)

		# for label, [x, y] in positiondict.items():
		# 	item_match_result = re.match(self.landmark_match, label)
		# 	position_row_table[item_match_result.group(1)].append(y)

		# position_row_table = {item[0]: item[1] for item in
		#                       sorted(position_row_table.items(), key=lambda record: record[0], reverse=False)}
		# position_row_temp = {}

		return positiondict, success

	# ------------------------------------------------
	# 名称：  position_loss
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
	def position_loss(self, best_four_label_choose, best_six_label_choose, find, level_four_min_loss,
	                  level_six_min_loss):
		loss = {}  # 闭合层级：丢失角点个数
		# 计算每个地标的定位识别状态
		for roi_item in self.rois:
			if "_R" in roi_item.label:
				continue
			loss_info = {}
			for level in ['4', '6']:
				label = roi_item.label
				if level == '4':
					candidate_landmarks = self.corners_levelfour(label)
					if candidate_landmarks is None:
						continue
					else:
						point1, point2, point3, point4 = candidate_landmarks

						loss_info['4'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
						if loss_info['4'] < level_four_min_loss:
							best_four_label_choose = roi_item.label
							level_four_min_loss = loss_info['4']

				elif level == '6':
					candidate_landmarks = self.corners_levelsix(label)
					if candidate_landmarks is None:
						continue
					else:
						point1, point2, point3, point4 = candidate_landmarks
						loss_info['6'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
						if loss_info['6'] < level_six_min_loss:
							best_six_label_choose = roi_item.label
							level_six_min_loss = loss_info['6']

				loss[label] = loss_info
		return best_four_label_choose, best_six_label_choose, loss

	# ------------------------------------------------
	# 名称：  position_landmark
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
	def position_landmark(self, image):
		self.ALL_LANDMARKS_DICT.clear()
		self.ALL_POSITIONS.clear()
		del self.rois
		# start = time.perf_counter()
		rows, cols, channels = image.shape
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			dest = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
		else:
			dest = image

		self.candidate_landmarks(dest, left_start=land_mark_left_start, left_end=land_mark_left_end,
		                         right_start=land_mark_right_start, right_end=land_mark_right_end)

		exception_labels = []
		if len(self.ALL_LANDMARKS_DICT.items()) > 0:
			for label, landmarkobj in self.ALL_LANDMARKS_DICT.items():
				label_match_result = re.match(self.landmark_match, label)
				if label_match_result is None: continue
				no = label_match_result.group(1)

				for landmarkname, p_landmark in self.ALL_LANDMARKS_DICT.items():
					if landmarkname == label: continue
					maybe = False
					if (re.match(self.landmark_match, landmarkname).group(
							1) < no and p_landmark.row > landmarkobj.row) or (
							re.match(self.landmark_match, landmarkname).group(
								1) > no and p_landmark.row < landmarkobj.row):
						maybe = True
					item_opposite_label = self.get_opposite_landmark(landmarkname)
					if item_opposite_label not in self.ALL_LANDMARKS_DICT and maybe == True:
						exception_labels.append(landmarkname)

					if item_opposite_label in self.ALL_LANDMARKS_DICT:
						item_opposite = self.ALL_LANDMARKS_DICT[item_opposite_label]
						if abs(int(item_opposite.row) - int(p_landmark.row)) > 100 and maybe == True:
							exception_labels.append(landmarkname)

		if exception_labels is not None and len(exception_labels) > 0:
			for lb in exception_labels:

				opposite_label = self.get_opposite_landmark(lb)

				lb_nbs = self.calc_neighbours(label_name=lb, ALL_LANDMARK_DIC=self.ALL_LANDMARKS_DICT)
				for_cols = list(filter(lambda item: item[0] == 'for_col', lb_nbs))

				if for_cols is None or len(for_cols) == 0 or opposite_label not in self.ALL_LANDMARKS_DICT: continue

				modify_status = True
				x, y = 0, 0
				ref_opposite_landmark = self.ALL_LANDMARKS_DICT[opposite_label]
				y = ref_opposite_landmark.row
				for ignore_lb, label_name in for_cols:
					if label_name in self.ALL_LANDMARKS_DICT:
						temp_obj = self.ALL_LANDMARKS_DICT[label_name]
						x = temp_obj.col
						break
				else:
					modify_status = False
					if lb in self.ALL_POSITIONS: self.ALL_POSITIONS.pop(lb)
					if lb in self.ALL_LANDMARKS_DICT: self.ALL_LANDMARKS_DICT.pop(lb)

				if modify_status == True:
					missobj = self.ALL_LANDMARKS_DICT[lb]
					missobj.row, missobj.col = y, x

			exception_labels.clear()

		if len(self.ALL_LANDMARKS_DICT.keys()) < 3:
			return dest, False

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}

		for landmark_roi in self.rois:
			landmark = landmark_roi.landmark
			if landmark is None or landmark_roi.label not in self.ALL_LANDMARKS_DICT:
				continue

			col = landmark.col
			row = landmark.row
			real_col, real_row = real_position_dic[landmark_roi.label]
			cv2.rectangle(dest, (col, row), (col + landmark.width, row + landmark.height), color=(255, 0, 255),
			              thickness=2)
			cv2.putText(dest,
			            "({},{})".format(real_col, real_row),
			            (col, row + 90),
			            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
			cv2.putText(dest,
			            "{}".format(landmark_roi.label),
			            (col, row + 60),
			            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
		position_dic, success = self.choose_best_cornors()

		if success == True:
			dest, success = self.__perspective_transform(dest, position_dic)

		return dest, success

	@property
	def rois(self):
		if self._rois is None or len(self._rois) == 0:
			landmark_rois = [
				LandMarkRoi(img=cv2.imread(os.path.join(ROIS_DIR, roi_img)), label=roi_img.split('.')[0].strip(), id=1)
				for
				roi_img in
				os.listdir(ROIS_DIR)]
			self._rois = landmark_rois
		return self._rois

	@rois.deleter
	def rois(self):
		if self._rois is not None:
			self._rois.clear()

	# 绘制网格
	def draw_grid_lines(self, img):
		H_rows, W_cols = img.shape[:2]
		for row in range(0, H_rows):
			if row % 100 == 0:
				cv2.line(img, (0, row), (W_cols, row), color=(255, 255, 0), thickness=1, lineType=cv2.LINE_8)
		for col in range(0, W_cols):
			if col % 50 == 0:
				cv2.line(img, (col, 0), (col, H_rows), color=(255, 255, 0), thickness=1, lineType=cv2.LINE_8)

	# ------------------------------------------------
	# 名称：  __perspective_transform
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def __perspective_transform(self, src, position_dic):
		H_rows, W_cols = src.shape[:2]
		# print(H_rows, W_cols)
		detected_landmarks = len(position_dic.items())

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}

		if position_dic is None or detected_landmarks < 3:
			mylog_error("检测到的地标小于三个，无法使用")
			return src, False

		left_points = []
		right_points = []
		for label, [x, y] in position_dic.items():
			if "L" in label:
				left_points.append((label, [x, y]))
			else:
				right_points.append((label, [x, y]))
		left_points.sort(key=lambda point: point[1][1])
		right_points.sort(key=lambda point: point[1][1])

		try:
			p1, p2, p3, p4 = left_points[0][1], right_points[0][1], left_points[1][1], right_points[1][1]
		except:
			return src, False
		else:
			pts1 = np.float32([p1, p3, p2, p4])
			pts2 = np.float32([real_position_dic.get(left_points[0][0]), real_position_dic.get(left_points[1][0]),
			                   real_position_dic.get(right_points[0][0]), real_position_dic.get(right_points[1][0])])

			M = cv2.getPerspectiveTransform(pts1, pts2)
			dst = cv2.warpPerspective(src, M, (H_rows, W_cols))
		return dst, True

	# ------------------------------------------------
	# 名称：  find_landmark
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def find_landmark(self, landmark_roi: LandMarkRoi, slide_window_obj: NearLandMark):
		if slide_window_obj is None: return
		col, row, slide_img = slide_window_obj.positioninfo
		roi = cv2.resize(landmark_roi.roi, (slide_window_obj.width, slide_window_obj.height))
		similar_rgb = self.__compare_rgb_similar(roi, slide_img)
		hsv_similar = self.__compare_hsv_similar(roi, slide_img)
		slide_window_obj.similarity = max(similar_rgb, hsv_similar)
		slide_window_obj.land_name = landmark_roi.label
		if similar_rgb >= 0.5 or hsv_similar > 0.5:

			landmark_roi.set_match_obj(slide_window_obj)
			self.ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj
			fail_time = 0
		else:
			for label, exist_land in self.ALL_LANDMARKS_DICT.items():
				if exist_land.col == slide_window_obj.col and exist_land.row == slide_window_obj.row: break
			else:
				if landmark_roi.landmark is None:
					# now  i have try my best to decide if  put in
					# may be another feature may work
					landmark_roi.set_match_obj(slide_window_obj)
					self.ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj

	def __landmark_position_dic(self):
		with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
			real_positions = pickle.load(coordinate)
		return real_positions

	def mk(self, target=None, left_open_mask=None):
		# ws, hs = [], []
		outer_width = outer_height = 5

		def inner_islandmark(c, color_code):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			# if len(ws) < 3 or len(hs) < 3:
			if not 7 < w < 19 or not 20 < h < 35: return False
			if not 100 < area < 450: return False
			target_roi = target[y - outer_width:y + h + outer_height,
			             x - outer_width:x + w + outer_width, :]

			# cv2.imshow("roi",target_roi)
			# cv2.waitKey(5000)
			if not np.any(target_roi): return False
			# img1_h, img1_w = target_roi.shape[0], target_roi.shape[1]
			# if img1_w < 5 and img1_h < 7: return False

			category = MvSuply.CATEGORY_CODE(
				target_roi)

			# targetroi_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
			# value = np.average(targetroi_gray)
			# print("color:{},category:{},area:{},width:{},height:{}".format(color_code, category,area,w,h))
			if category == 0 and color_code == "BLUE": return False

			return True

		global LANDMARK_COLOR_INFO, COLOR_RANGE
		choosed_contours = []
		if "YELLOW" in COLOR_RANGE: COLOR_RANGE.pop('YELLOW')
		for color_code, [low, high] in COLOR_RANGE.items():
			foreground = self.get_colorrange_binary(color_code, target, low, high)

			# foreground = cv2.medianBlur(foreground, 3)

			disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			foreground = cv2.filter2D(foreground, -1, disc)
			# cv2.imshow(color_code, foreground)
			foreground = cv2.bitwise_and(foreground, foreground, mask=left_open_mask)
			ret, foreground = cv2.threshold(foreground, LANDMARK_THREHOLD_START,
			                                LANDMARK_THREHOLD_END,
			                                cv2.THRESH_BINARY)  # 110,255


			contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			contours = sorted(list(filter(lambda c: inner_islandmark(c, color_code), contours)),
			                  key=lambda c: cv2.contourArea(c),
			                  reverse=True)


			if contours is not None and len(contours) > 0:
				if color_code == "BLUE":
					col, row, w, h = cv2.boundingRect(contours[0])
					choosed_contours.append([color_code, contours[0], row])
				else:
					for c in contours:
						col, row, w, h = cv2.boundingRect(c)
						# print(color_code, row)
						choosed_contours.append([color_code, c, row])

		choosed_contours = sorted(choosed_contours, key=lambda infos: infos[2])
		COLOR_INPUT = [colorinfo[0] for colorinfo in choosed_contours]
		find_color_num = len(COLOR_INPUT)
		if find_color_num < 2:
			return (None, None)

		left_landmarks = list(filter(lambda color: 'L' in color[0][0:5], LANDMARK_COLOR_INFO.items()))
		left_landmarks = sorted(left_landmarks, key=lambda item: item[0][2])
		# print(left_landmarks)

		start = 0
		current_landmark_names = []
		current_nos = []
		find = False
		while start <= len(left_landmarks):

			choosed_landmarks = [item for item in left_landmarks[start:]]
			if len(choosed_landmarks) < 2 or len(COLOR_INPUT) < 2: break

			for index in range(min(find_color_num, len(choosed_landmarks))):

				if COLOR_INPUT[index] != choosed_landmarks[0:find_color_num][index][1]:
					break
			else:
				find = True
				for i in range(min(find_color_num, len(choosed_landmarks))):
					current_nos.extend(
						[choosed_landmarks[i][0][2:3]])
					current_landmark_names.extend(
						[choosed_landmarks[i][0][0:5], self.get_opposite_landmark(choosed_landmarks[i][0][0:5])])

			start += 1

			if find == True:
				break

		# print(current_landmark_names)
		return (current_landmark_names, current_nos) if find else (None, None)

	# ------------------------------------------------
	# 名称：  candidate_landmarks
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def candidate_landmarks(self, dest=None, left_start=110, left_end=210,
	                        right_start=510, right_end=600):
		global rows, cols, step
		target = dest

		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		img_world = np.ones_like(gray)
		ret, img_world = cv2.threshold(img_world, 0, 255, cv2.THRESH_BINARY)

		def set_mask_area(x: int, y: int, width: int, height: int):
			img_world[y:y + height, x:x + width] = 0

		left_open_mask = np.zeros_like(gray)
		left_open_mask[:, left_start:left_end] = 255

		right_open_mask = np.zeros_like(gray)
		right_open_mask[:, right_start:right_end] = 255

		bigest_h, bigest_w = 0, 0

		self.checkmatch_roi_contour(bigest_h, bigest_w, img_world, left_open_mask, right_open_mask,
		                            set_mask_area, target)

	def checkmatch_roi_contour(self, bigest_h, bigest_w, img_world, left_open_mask, right_open_mask,
	                           set_mask_area, target):

		a, b = self.mk(target, left_open_mask)  # 0.106s
		if a is None:
			return None
		else:

			select_rois = filter(lambda roi: roi.landmark is None and roi.label.strip() in a, self.rois)

		for roi_template in select_rois:
			color_codes = [color_code_info for color_label, color_code_info in LANDMARK_COLOR_INFO.items() if
			               roi_template.label in color_label]
			color_code = None if len(color_codes) == 0 else color_codes[0]
			if color_code is None: continue
			contours = self.landmark_foreground_method1(
				left_open_mask,
				right_open_mask,
				roi_template,
				target)
			if contours is None or len(contours) == 0:
				continue
			c = contours[0]
			center_x, center_y = self.calc_landmarkcenter(c)
			rect = cv2.boundingRect(c)
			x, y, w, h = rect
			if h > bigest_h: bigest_h = h
			if w > bigest_w: bigest_w = w

			neighbours = self.calc_neighbours(roi_template)
			self.init_all_landmarks_dict(bigest_h, bigest_w, c, center_x, center_y, color_code, contours, h, neighbours,
			                             roi_template, set_mask_area, target, w, x, y)


	def landmark_foreground_method1(self, left_open_mask, right_open_mask,
	                                roi_template, target):
		outer_width = outer_height = 5

		def warp_filter(c, label):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			if w < 4 and h < 4: return False
			if w > 50 or h > 50: return False
			if area < 100: return False
			target_roi = target[y - outer_width:y + h + outer_height,
			             x - outer_width:x + w + outer_width, :]

			if not np.any(target_roi): return False

			category = MvSuply.CATEGORY_CODE(
				target_roi)
			if category == 1: return True
			# if w > 50 or h > 50 or w < 3 or h < 3:
			# 	return False
			return False

		# target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		# ret, light_binary = cv2.threshold(target_gray, 30, 170,
		#                                   cv2.THRESH_BINARY)  # 110,255

		# value = np.average(targetroi_gray)
		# 检测二值图像
		foreground = MvSuply.FIND_IT(target, roi_template.roi)
		foreground = cv2.bitwise_and(foreground, foreground, mask=left_open_mask if roi_template.label.find(
			"L") > 0 else right_open_mask)
		foreground = cv2.medianBlur(foreground, 5)

		# foreground = cv2.bitwise_and(foreground, foreground, mask=img_world)
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foreground = cv2.dilate(foreground, kernel)
		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)
		ret, foreground = cv2.threshold(foreground, LANDMARK_THREHOLD_START, LANDMARK_THREHOLD_END,
		                                cv2.THRESH_BINARY)  # 110,255

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		contours = list(filter(lambda c: warp_filter(c, roi_template.label), contours)) if len(
			contours) > 1 else contours
		contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
		return contours

	# 地标检测方法2
	def landmark_foreground_method2(self, left_open_mask, right_open_mask,
	                                roi_template, target, color_code):
		def warp_filter(c, label):
			x, y, w, h = cv2.boundingRect(c)
			# isbig = 100 <= cv2.contourArea(c) < 3600
			# if label == 'NO3_L': print("landmark:x:{} ,y:{} ,w:{},h:{}".format(x, y, w, h))
			if w > 20 or h > 20 or w < 3 or h < 3:
				return False
			return True

		foreground = self.get_colorrange_binary(color_code, target)

		foreground = cv2.bitwise_and(foreground, foreground, mask=left_open_mask) if roi_template.label.find(
			"L") > 0 else \
			cv2.bitwise_and(foreground, foreground, mask=right_open_mask)
		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)
		ret, foreground = cv2.threshold(foreground, LANDMARK_THREHOLD_START, LANDMARK_THREHOLD_END,
		                                cv2.THRESH_BINARY)  # 110,255

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = list(filter(lambda c: warp_filter(c, roi_template.label), contours)) if len(
			contours) > 1 else contours

		contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
		return foreground, contours

	# 根据颜色区域获取二值图像
	def get_colorrange_binary(self, color_code=None, target=None, color_low=None, color_high=None):

		foreground = None
		b, g, r = cv2.split(target)
		if color_code == 'GREEN':
			gx_ignore, gy_ignore = np.where((b > 100) | (r > 100))
			g[gx_ignore, gy_ignore] = 0
			ret, g = cv2.threshold(g, 140,
			                       255,
			                       cv2.THRESH_BINARY)  # 110,255
			# g = cv2.bitwise_and(g, g, mask=foreground)
			foreground = g

		elif color_code == 'RED':
			rx_ignore, ry_ignore = np.where((g > 100) | (b > 100))
			r[rx_ignore, ry_ignore] = 0
			ret, r = cv2.threshold(r, 140,
			                       255,
			                       cv2.THRESH_BINARY)  # 110,255

			foreground = r

		elif color_code == 'BLUE':
			bx_ignore, by_ignore = np.where((g > 100) | (r > 100))
			b[bx_ignore, by_ignore] = 0
			ret, b = cv2.threshold(b, 140,
			                       255,
			                       cv2.THRESH_BINARY)  # 110,255

			# b = cv2.bitwise_and(b, b, mask=foreground)

			foreground = b
		else:
			hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
			if color_code is not None:
				color_low, color_high = COLOR_RANGE[color_code]
			color1_min, color1_max = np.array(color_low), np.array(color_high)
			foreground = cv2.inRange(hsv, color1_min, color1_max)

		return foreground

	def delete_error_landmark(self):
		def no_direct(label):
			result = re.match(landmark_pattern, label)
			return int(result.group(1)), result.group("direct")

		will_delete = []
		for label, landmark_item in self.ALL_LANDMARKS_DICT.items():
			no, direct = no_direct(label)
			current_row, current_col = landmark_item.row, landmark_item.col

			if no == 1:
				continue
			prev = no - 1
			error = False
			while prev > 1:
				prev_label = "NO{}_{}".format(prev, direct)
				if prev_label in self.ALL_LANDMARKS_DICT:
					row, col = self.ALL_LANDMARKS_DICT[prev_label].row, self.ALL_LANDMARKS_DICT[prev_label].col
					if current_row > row:
						error = False
						break
				else:
					prev -= 1

			if error == True:
				will_delete.append(label)

		for label in will_delete:
			del self.ALL_LANDMARKS_DICT[label]

	# 初始化所有地标
	def init_all_landmarks_dict(self, bigest_h, bigest_w, c, center_x, center_y, color_code, contours, h, neighbours,
	                            roi_template, set_mask_area, target, w, x, y):
		match_ok = self.color_match_result(target, color_code, c)
		if match_ok == False: return

		for flag, ref_label in neighbours:

			if flag == 'for_col' and ref_label in self.ALL_LANDMARKS_DICT:
				ref_landmark = self.ALL_LANDMARKS_DICT[ref_label]
				if abs(ref_landmark.col - x) <= 50:
					landmark_obj = NearLandMark(x, y,
					                            target[y:y + h, x:x + w])
					landmark_obj.width = max(bigest_w, w)
					landmark_obj.height = max(bigest_h, h)
					set_mask_area(center_x - 50, center_y - 50, 200, 200)
					landmark_obj.add_maybe_label(roi_template.label)
					roi_template.set_match_obj(landmark_obj, target)
					self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj
					break
			elif flag == 'for_row' and ref_label in self.ALL_LANDMARKS_DICT:
				ref_landmark = self.ALL_LANDMARKS_DICT[ref_label]
				if abs(ref_landmark.row - y) <= 50:
					landmark_obj = NearLandMark(x, y,
					                            target[y:y + h, x:x + w])
					landmark_obj.width = max(bigest_w, w)
					landmark_obj.height = max(bigest_h, h)
					set_mask_area(center_x - 50, center_y - 50, 200, 200)
					landmark_obj.add_maybe_label(roi_template.label)
					roi_template.set_match_obj(landmark_obj, target)
					self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj
					break
		else:
			for key, color_code in LANDMARK_COLOR_INFO.items():
				if roi_template.label in key:
					match_ok = self.color_match_result(target, color_code, c)
					if match_ok:
						rect = cv2.boundingRect(c)
						best_x, best_y, best_w, best_h = rect
						landmark_obj = NearLandMark(best_x, best_y,
						                            target[best_y:best_y + best_h, best_x:best_x + best_w])
						landmark_obj.width = max(bigest_w, best_w)
						landmark_obj.height = max(bigest_h, best_h)
						landmark_obj.add_maybe_label(roi_template.label)
						if len(contours) == 1:
							set_mask_area(center_x - 50, center_y - 50, 200, 200)
						roi_template.set_match_obj(landmark_obj, target)
						self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj
						break

	def init_landmark_showwindow(self, NO1_L, NO1_R, NO2_L, NO2_R, NO3_L, NO3_R, foreground, roi_template):
		if DEBUG == True:
			if roi_template.label == "NO1_L":
				NO1_L = foreground
			elif roi_template.label == "NO2_L":
				NO2_L = foreground
			elif roi_template.label == "NO3_L":
				NO3_L = foreground
			elif roi_template.label == "NO1_R":
				NO1_R = foreground
			elif roi_template.label == "NO2_R":
				NO2_R = foreground
			elif roi_template.label == "NO3_R":
				NO3_R = foreground
		return NO1_L, NO1_R, NO2_L, NO2_R, NO3_L, NO3_R

	def show_landmark_binary(self, GREEN=None, RED=None, BLUE=None):
		if DEBUG == True:


			HEJI_IMG = np.vstack([GREEN, RED, BLUE])
			HEJI_IMG = cv2.resize(HEJI_IMG, (700, 800))

			cv2.imshow("HEJI_IMG", HEJI_IMG)

	def calc_neighbours(self, roi_template=None, label_name=None, ALL_LANDMARK_DIC=None):

		lb = None

		if roi_template is not None:
			lb = roi_template.label

			neighbours = [('for_row', self.get_opposite_landmark(roi_template.label)),
			              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=1)),
			              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=2)),
			              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=-1)),
			              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=-2))]
		if label_name is not None:
			lb = label_name
			neighbours = [('for_row', self.get_opposite_landmark(label_name)),
			              ('for_col', self.__fetch_neigbbour(label_name, sourth_step=0, west_step=1)),
			              ('for_col', self.__fetch_neigbbour(label_name, sourth_step=0, west_step=2)),
			              ('for_col', self.__fetch_neigbbour(label_name, sourth_step=0, west_step=-1)),
			              ('for_col', self.__fetch_neigbbour(label_name, sourth_step=0, west_step=-2))]

		if ALL_LANDMARK_DIC is not None:
			result = re.match(self.landmark_match, lb)
			current_d = result.group(2)
			neighbours.extend([('for_col', name) for name, obj_land in ALL_LANDMARK_DIC.items() if
			                   "_{}".format(current_d) in name and ('for_col', name) not in neighbours])

		return neighbours

	def calc_landmarkcenter(self, c):
		area = cv2.contourArea(c)
		rect = cv2.boundingRect(c)
		x, y, w, h = rect
		M = cv2.moments(c)
		try:
			center_x = int(M["m10"] / M["m00"])
			center_y = int(M["m01"] / M["m00"])
		except:
			center_x, center_y = x, y
		return center_x, center_y


# ------------------------------------------------
# 名称：LasterDetector
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class LasterDetector(BaseDetector):

	def __init__(self):
		super().__init__()
		self.laster = None

	def location_laster(self, img_show, img_copy, middle_start=120, middle_end=450):

		def __filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			# logger("laster is {}".format(area), 'info')

			if laster_min_width < w < laster_max_width and laster_min_height < h < laster_max_height and \
					laster_min_area < area < laster_max_area:
				return True
			else:
				return False

		foregroud, contours = self.green_contours(img_copy, middle_start, middle_end)
		# cv2.imshow("green", foregroud)
		# contours = list(filter(__filter_laster_contour, contours))

		if contours is None or len(contours) == 0 or len(contours) > 1:
			return None, foregroud

		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)

		try:
			self.laster = Laster(contours[0], id=0)
			self.laster.modify_box_content()
		except Exception as e:
			mylog_error("laster contour is miss")
			raise e

		return self.laster, foregroud


# ------------------------------------------------
# 名称：HockDetector
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class HockDetector(BaseDetector):

	def __init__(self):
		super().__init__()
		self.hock = None
		self._roi = None
		self.mask = None
		self.has_init = False
		self.has_update = False
		self.has_stable = False

	def load_or_update_position(self, p_tu=None, hock_contour_num=0):
		if self.has_init == False:
			self.f = f()
			self.f.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
			self.f.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
			self.f.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
			self.f.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
			self.has_init = True

		if p_tu is not None:
			x, y = self.get_predict()
			self.has_stable = hock_contour_num == 1 and abs(x - p_tu[0]) < 100 and abs(
				y - p_tu[1]) < 100
			pos = np.array([*p_tu], dtype=np.float32)
			mes = np.reshape(pos, (2, 1))
			self.f.correct(mes)
			self.has_update = True

	def get_predict(self):
		guass_position = self.f.predict()
		x, y = int(guass_position[0][0]), int(guass_position[1][0])
		return x, y

	# ------------------------------------------------
	# 名称：find_edige
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_edige(self, dest):

		bottom_edge = 0
		for row in range(IMG_HEIGHT // 2, IMG_HEIGHT):
			value = sum([dest[row, i, 0] for i in range(IMG_WIDTH // 2, IMG_WIDTH)])
			if value == 0:
				# print("{} 为0".format(row))
				bottom_edge = row
				break

		right_edge = 0
		for col in range(IMG_WIDTH // 2, IMG_WIDTH):
			value = sum([dest[i, col, 0] for i in range(IMG_HEIGHT // 2, IMG_HEIGHT)])
			if value == 0:
				right_edge = value
				break

		return bottom_edge, right_edge

	# ------------------------------------------------
	# 名称：find_green_contours
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_green_contours(self, img, middle_start=120, middle_end=450):
		foreground, contours = self.green_contours(img, middle_start=middle_start, middle_end=middle_end)
		return foreground, contours

	@property
	def hock_roi(self):
		if self._roi is None:
			hock_rois = [
				HockRoi(img=cv2.imread(os.path.join(HOCK_ROI, roi_img)))
				for
				roi_img in
				os.listdir(HOCK_ROI)]
			if len(os.listdir(HOCK_ROI)) == 0:
				self._roi = None
			self._roi = hock_rois[0]
		return self._roi

	# ------------------------------------------------
	# 名称：hock_foreground
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def hock_foreground(self, img_copy, middle_start=110, middle_end=500):
		# target_hsvt = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
		# img_roi_hsvt = cv2.cvtColor(self.hock_roi.img, cv2.COLOR_BGR2HSV)

		# 检测定位钩方法
		foreground, _d = self.yellow_contours(img_copy)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.dilate(foreground, kernel)

		foreground[0:, 0:middle_start] = 0
		foreground[0:, middle_end:] = 0

		ret, foreground = cv2.threshold(foreground, 20, 255, cv2.THRESH_BINARY)

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return contours, foreground

	# ------------------------------------------------
	# 名称：location_hock_withlandmark
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_hock_withlandmark(self, img_show, img_copy, laster_of_on=True, middle_start=120,
	                               middle_end=470):
		def filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)

			if BIG_OR_SMALL_LASTER==1:
				# print("small area:{},x:{},y:{},w:{},h:{}".format(area, x, y, w, h))
				if w < smalllaster_min_width and h < smalllaster_min_height and area<smalllaster_min_area:
					return False
			else:
				# print("big area:{},x:{},y:{},w:{},h:{}".format(area, x, y, w, h))
				if w < biglaster_min_width and h < biglaster_min_height and area < biglaster_min_area:
					return False

			if x < middle_start or x > middle_end:
				return False

			return True

		if laster_of_on == True:
			foregroud, laster_contours = self.green_contours(img_copy, middle_start, middle_end)
			laster_contours = list(filter(filter_laster_contour, laster_contours))
			laster_contours = sorted(laster_contours, key=lambda c: cv2.contourArea(c), reverse=False)
			biggest_c = None
			if laster_contours is not None and len(laster_contours) > 0:

				if len(laster_contours) > 1:
					biggest_c = laster_contours[0]
					max_area = 0

					# mask_label=np.zeros_like(gray)
					x_list, y_list = [], []

					biggest_c = self.find_biggest_hockcontour(biggest_c, laster_contours, max_area)

					b_x, b_y, b_w, b_h = cv2.boundingRect(biggest_c)
					M_biggest = cv2.moments(biggest_c)
					biggest_cx = int(M_biggest['m01'] / M_biggest['m00'])
					biggest_cy = int(M_biggest['m01'] / M_biggest['m00'])

					for c in laster_contours:
						m = cv2.moments(biggest_c)
						cx = int(m['m01'] / m['m00'])
						cy = int(m['m01'] / m['m00'])
						if abs(cx - biggest_cx) > 100 or abs(cy - biggest_cy) > 100: continue
						x, y, w, h = cv2.boundingRect(c)
						# 做人建议正直点，有幸看到这里也是有缘人；不要因小事情，影响自己格局，不值当
						x_list.append(x)
						x_list.append(x + w)
						y_list.append(y)
						y_list.append(y + h)

					min_x, min_y = min(x_list), min(y_list)
					max_x, max_y = max(x_list), max(y_list)

					cent_x = int(0.5 * (min_x + max_x))
					cent_y = int(0.5 * (min_y + max_y))
					try:

						self.hock = Hock(biggest_c)
						if self.has_stable == True:
							x, y = self.get_predict()
							if abs(x - cent_x) > 100 or abs(y - cent_y) > 100:
								self.hock.set_position(x, y)
							# self.load_or_update_position((cent_x, cent_y))
							else:
								self.hock.set_position(cent_x, cent_y)
								self.load_or_update_position((cent_x, cent_y), len(laster_contours))
						# self.hock.set_position(cent_x, cent_y)
						# self.load_or_update_position((cent_x, cent_y))
						self.hock.modify_box_content()
					# cv2.rectangle(img_show, (min_x, min_y), (max_x, max_y),
					#           (30, 144, 255), 3)

					except Exception as e:
						print(e.__str__())
						raise e
				elif len(laster_contours) == 1:
					biggest_c = laster_contours[0]
					# cv2.drawContours(img_show, laster_contours, -1, (255, 0, 255), 3)
					# 让我干半个多月了，只给一个项目，不是坑不给我，我也是心累啊
					try:
						self.hock = Hock(biggest_c)

						if self.has_stable == True:
							x, y = self.get_predict()
							if abs(x - self.hock.center_x) > 100 or abs(y - self.hock.center_y) > 100:
								self.hock.set_position(x, y)
							# self.load_or_update_position((x, y))
							else:
								self.load_or_update_position((self.hock.center_x, self.hock.center_y), 1)
						else:
							self.load_or_update_position((self.hock.center_x, self.hock.center_y))

						self.hock.modify_box_content()

					# cv2.drawContours(img_show, laster_contours, -1, (255, 0, 255), 3)

					except Exception as e:
						print(e.__str__())
						raise e
				else:
					plchandle = PlcHandle()
					plchandle.laster = 0
					plchandle.biglaster = 1

					try:
						if self.has_stable == True:
							guass_position = self.get_predict()
							self.hock = Hock(biggest_c)
							self.hock.set_position(*guass_position)
							self.hock.modify_box_content()
					except Exception as e:
						print(e.__str__())
						raise e

		else:
			assert laster_of_on, "没有开启激光灯"

		return self.hock, foregroud

	def find_biggest_hockcontour(self, biggest_c, laster_contours, max_area):
		for c in laster_contours:
			area = cv2.contourArea(c)
			if area > max_area:
				max_area = area
				biggest_c = c
		return biggest_c

	# ------------------------------------------------
	# 名称：location_hock_withoutlandmark
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_hock_withoutlandmark(self, img_show, middle_start=middle_start_withoutlandmark,
	                                  middle_end=middle_end_withoutlandmark):

		# def if_rectangle(c):
		# 	peri = cv2.arcLength(c, True)
		# 	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		# 	return len(approx) == 4
		# img2_points = self.feature_similar_radio(img_show)

		# print(len(img2_points))
		# print(img2_points)

		def filter_hock_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)

			if x < laster_min_area or x > laster_max_area:
				return False

			if area < laster_min_area or area > laster_max_area:
				return False

			# if feature_radio < 3:
			# 	return False

			# print("wide is {},height is {}".format(w, h))
			return True
		# 调部门感觉有套路在里面，某某人自保吧，哎
		contours, foregroud = self.hock_foreground(img_show, middle_start, middle_end)
		# green_ret, foreground = cv2.threshold(foregroud, 40, 255, cv2.THRESH_BINARY)
		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foregroud = cv2.filter2D(foregroud, -1, disc)

		contours = list(filter(filter_hock_contour, contours))

		# print("after filter contours is {}".format(len(contours)))
		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)
		if contours is None or len(contours) == 0:  # or len(contours) > 1
			return None, foregroud

		hock = None
		try:
			hock = Hock(contours[0])
			hock.modify_box_content()
			cv2.putText(img_show, hock.box_content,
			            (hock.boxcenterpoint[0], hock.boxcenterpoint[1]),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		except Exception as e:
			mylog_error("hock contour is miss")

		return hock, foregroud

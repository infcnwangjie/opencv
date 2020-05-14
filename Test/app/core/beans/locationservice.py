# -*- coding: utf-8 -*-
import math
import random

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from app.config import SDK_OPEN, DISTANCE_SAMEXLANDMARK_SPACE, DISTANCE_SAMEYLANDMARK_SPACE
from app.core.exceptions.allexception import NotFoundBagException, NotFoundHockException
from app.core.processers.bag_detector import BagDetector
from app.core.processers.laster_detector import LasterDetector
from app.core.processers.preprocess import Preprocess
from app.core.beans.models import Box, Bag, Laster, Hock
from app.log.logtool import mylog_debug, mylog_error
from app.status import Landmark_Model_Select

BAG_AND_LANDMARK = 0
ONLY_BAG = 1
ONLY_LANDMARK = 2
ALL = 3


class PointLocationService:
	'''已废弃'''
	def __init__(self, img=None, print_or_no=True):
		self._img = img
		self.bags = []  # 识别出来的袋子
		self.landmarks = []  # 识别出来的地标
		self.hock = None  # 识别出来的钩子
		self.laster = None  # 识别出来的激光灯
		self.nearestbag = None  # 与钩子最近的袋子
		self.landmark_select_method = None  # 筛选地标的方法
		self.landmarkvirtualdistance = None  # 虚拟地标的图像距离

	@property
	def shape(self):
		gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	@property
	def img(self):
		return self._img

	@img.setter
	def img(self, value):
		self._img = value

	def __enter__(self):
		return self

	# 计算袋的坐标
	def computer_bags_location(self, digitdetector=None):
		# process = Preprocess(self.img)
		process = BagDetector(self.img)
		self.bags = process.processed_bag

	# # 用紫色画轮廓
	# cv2.drawContours(self.img, moderatesize_countours, -1, (0, 255, 255), 3)

	# 计算地标的坐标
	def computer_landmarks_location(self):
		processor = LandMarkDetector(self.img)
		self.landmarks = processor.processedlandmarkimg

	def find_nearest_bag(self):
		'''找到离钩子最近的袋子'''
		self.computer_landmarks_location()  # 计算地标位置
		self.computer_bags_location()  # 计算所有袋子定位
		if self.bags is None or len(self.bags) == 0:
			mylog_debug("没有发现袋子，请核查实际情况！")
			return None

		hockposition = self.compute_hook_location  # 计算钩子位置，为了移动钩子

		if hockposition is None:
			mylog_error("没有发现钩子，请核查实际情况！")
			return None

		distance_dict = {}
		for bag in self.bags:
			img_distance, real_distance, _movex, _movey = self.compute_distance(bag.boxcenterpoint, hockposition)
			distance_dict[str(int(img_distance))] = bag

		smallestindex = min(distance_dict.keys(), key=lambda item: int(item))
		nearest_bag = distance_dict[str(smallestindex)]
		# cv2.drawContours(self.img, [nearest_bag.contour], -1, (0, 255, 0), 1)

		self.nearestbag = nearest_bag  # 计算出离钩子距离最近的袋子

		img_distance, real_distance, move_x, move_y = self.compute_distance(nearest_bag.boxcenterpoint, hockposition)
		cv2.circle(self.img, nearest_bag.boxcenterpoint, 40, (0, 255, 0), thickness=3)
		cv2.putText(self.img, "nearest bag,distance is {} pixer".format(img_distance),
		            (nearest_bag.boxcenterpoint[0] + 50, nearest_bag.boxcenterpoint[1],), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		            (255, 0, 0), 2)
		if move_x < 0:
			moveinfo_x = u"move left:{}cm".format(
				abs(round(move_x, 2)))
		else:
			moveinfo_x = u"move right:{}cm".format(
				round(move_x, 2))

		if move_y > 0:
			moveinfo_y = u"move back:{}cm".format(
				round(move_y, 2))
		else:
			moveinfo_y = u"move forward:{}cm".format(
				abs(round(move_y, 2)))

		# # 袋子质心到钩子的直线
		# cv2.line(self.img, hockposition, nearest_bag.boxcenterpoint, (0, 255, 255), thickness=3)
		# # 袋子质心与钩子所在x轴的垂直线
		# cv2.line(self.img, nearest_bag.boxcenterpoint, (nearest_bag.boxcenterpoint[0], hockposition[1]),
		#          (0, 255, 255),
		#          thickness=3)

		# word_position = (int(bag.x - 60), hockposition[1] - 60)
		#
		# cv2.putText(self.img, moveinfo_y, word_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
		#
		# word_position_x = (int(bag.x + abs(0.5 * (bag.x - hockposition[0]))), hockposition[1] + 100)
		# cv2.putText(self.img, moveinfo_x, word_position_x, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

		# cv2.line(self.img, (nearest_bag.boxcenterpoint[0], hockposition[1]), hockposition, (0, 255, 255),
		#          thickness=3)
		mylog_debug(moveinfo_x + moveinfo_y)
		# return img_distance, real_distance, move_x, move_y
		return nearest_bag.boxcenterpoint, hockposition

	# 将坐标打印到图片上
	def print_location_onimg(self):
		# cv2.imshow('im', self.img)
		# cv2.namedWindow('im', cv2.WINDOW_KEEPRATIO)
		# cv2.imshow("im", self.img)
		return self.img

	def __exit__(self, exc_type, exc_val, exc_tb):
		del self.bags, self.laster, self.hock, self.landmarks

	@property
	def compute_hook_location(self):
		'''
		摄像头的位置只有X轴移动，在Y轴是固定的；激光灯理想的安装情况是，灯光的Y轴与钩子同步，跟随点击运动。
		已知量：灯光的位置、地标间隔实际距离、地标间隔图像距离，摄像头在Y轴上距离是固定的，不用去管
		:return:
		'''

		processor = LasterDetector(self.img)
		self.laster = processor.processed_laster
		lasterposition = self.laster.boxcenterpoint  # 激光灯位置X轴较钩子小
		hockposition = lasterposition  # 钩子的坐标
		cv2.putText(self.img, "hock->({},{})".format(hockposition[0], hockposition[1]),
		            (hockposition[0], hockposition[1]),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 0, 0), 2)

		return hockposition

	def compute_distance(self, point1, point2):
		'''
		:param point1:
		:param point2:
		:return: 两点之间图像像素距离，两点之间真实距离，
		两点之间X轴需要真实移动距离，两点之间Y轴需要真实移动距离
		'''
		if point1 is None or point2 is None:
			raise Exception("两个像素点不能为空")

		if self.landmarkvirtualdistance is None:
			# 寻找左侧的地标
			# self.computer_landmarks_location()  # 计算地标仅仅为了像素位置、实际位置的换算
			left_landmarks, right_landmarks = [], []
			grayimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
			rows, cols = grayimg.shape

			for landmark in self.landmarks:
				if landmark.x < cols * 0.3:
					left_landmarks.append(landmark)
				elif landmark.x > cols * 0.75:
					right_landmarks.append(landmark)
			# 选择挨着的两个地标
			left_landmarks = sorted(left_landmarks, key=lambda position: position.y)
			right_landmarks = sorted(right_landmarks, key=lambda position: position.y)

			if len(left_landmarks) >= 2:
				landmark1, landmark2 = left_landmarks[0:2]
				left_landmark_distance = abs(int(landmark1.y - landmark2.y))
				self.landmarkvirtualdistance = left_landmark_distance
				self.landmark_select_method = Landmark_Model_Select.CHOOSE_X_SAME
				mylog_debug("用的左边两个地标,地标图片距离:{}".format(left_landmark_distance))

			elif len(right_landmarks) >= 2:
				landmark1, landmark2 = right_landmarks[0:2]
				right_landmark_distance = abs(int(landmark1.y - landmark2.y))
				self.landmarkvirtualdistance = right_landmark_distance
				self.landmark_select_method = Landmark_Model_Select.CHOOSE_X_SAME
				mylog_debug("用的右边两个地标,地标图片距离:{}".format(right_landmark_distance))
			else:
				for left in left_landmarks:
					for right in right_landmarks:
						distance = abs(int(left.y - right.y))
						if distance < 20:
							self.landmark_select_method = Landmark_Model_Select.CHOOSE_Y_SAME
							self.landmarkvirtualdistance = abs(int(left.x - right.y))
							mylog_debug("用的左右两边的两个地标,地标图片距离:{}".format(self.landmarkvirtualdistance))
							break

		point1_x, point1_y = point1
		point2_x, point2_y = point2

		# 计算图像中两个点之间的像素距离
		img_distance = math.sqrt(
			math.pow(point2_y - point1_y, 2) + math.pow(point2_x - point1_x, 2))

		if self.landmarkvirtualdistance is None or self.landmarkvirtualdistance == 0:
			self.landmarkvirtualdistance = 1000000
		# 两个点之间的实际距离
		land_mark_realspace = DISTANCE_SAMEXLANDMARK_SPACE if self.landmark_select_method == Landmark_Model_Select.CHOOSE_X_SAME else DISTANCE_SAMEYLANDMARK_SPACE
		real_distance = round(img_distance * land_mark_realspace / self.landmarkvirtualdistance, 2)

		x_move = round(point1_x - point2_x)
		real_x_distance = round(x_move * land_mark_realspace / self.landmarkvirtualdistance, 2)

		y_move = round(point1_y - point2_y)
		real_y_distance = round(y_move * land_mark_realspace / self.landmarkvirtualdistance, 2)
		return img_distance, real_distance, real_x_distance, real_y_distance

# -*- coding: utf-8 -*-
import math
import random

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from app.config import SDK_OPEN, DISTANCE_SAMEXLANDMARK_SPACE, DISTANCE_SAMEYLANDMARK_SPACE
from app.core.exceptions.allexception import NotFoundBagException, NotFoundHockException
from app.core.preprocess.preprocess import Preprocess
from app.core.target_detect.models import Box, LandMark, Bag, Laster, Hock
from app.log.logtool import mylog_debug, mylog_error
from app.status import Landmark_Model_Select

BAG_AND_LANDMARK = 0
ONLY_BAG = 1
ONLY_LANDMARK = 2
ALL = 3


class PointLocationService:
	def __init__(self, img=None, print_or_no=True):
		self._img = img
		self.bags = []  # 识别出来的袋子
		self.landmarks = []  # 识别出来的地标
		self.hock = None  # 识别出来的钩子
		self.laster = None  # 识别出来的激光灯
		# self.processfinish = False  # 是否处理完成
		self.nearestbag = None  # 与钩子最近的袋子
		self.print_or_no = print_or_no  # 是否显示图片
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
	def inner_computer_bags_location(self, digitdetector=None):
		process = Preprocess(self.img)
		bag_binary, contours = process.processed_bag
		if contours is None or len(contours) == 0:
			return
		# 大小适中的轮廓，过小的轮廓被去除了
		moderatesize_countours = []
		boxindex = 0
		self.bags.clear()
		rows, cols = bag_binary.shape
		for countour in contours:
			countour_rect = cv2.boundingRect(countour)
			rect_x, rect_y, rect_w, rect_h = countour_rect

			center_x, center_y = (rect_x + round(rect_w * 0.5), rect_y + round(rect_h * 0.5))

			if cv2.contourArea(countour) > 500 and rect_h < 300 and 0.32 * cols < center_x < 0.72 * cols:
				moderatesize_countours.append(countour)
				box = Bag(countour, bag_binary, id=boxindex)
				boxindex += 1
				box.modify_box_content(digitdetector, no_num=True)
				cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0] + 50, box.boxcenterpoint[1] + 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
				self.bags.append(box)

		# 用紫色画轮廓
		cv2.drawContours(self.img, moderatesize_countours, -1, (0, 255, 0), 1)

	# 计算地标的坐标
	def iner_computer_landmarks_location(self, digitdetector=None):
		process = Preprocess(self.img)
		binary_image, contours = process.processedlandmarkimg
		if contours is None or len(contours) == 0:
			return

		rows, cols = binary_image.shape

		boxindex = 0
		for countour in contours:
			x, y, w, h = cv2.boundingRect(countour)
			cent_x, cent_y = x + round(w * 0.5), y + round(h * 0.5)
			if 0 < cent_x < 0.3 * cols or 0.75 * cols < cent_x < cols:
				box = LandMark(countour, binary_image, id=boxindex, digitdetector=digitdetector)
				box.modify_box_content(digitdetector, no_num=True)
				boxindex += 1
				self.landmarks.append(box)
		# 过滤袋子->将面积较小的袋子移除
		# self.filterlandmark()
		good_contours = []
		for box in self.landmarks:
			# 标记坐标和值
			cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0] + 50, box.boxcenterpoint[1] + 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
			good_contours.append(box.contour)

		if len(good_contours) > 0:
			# 用黄色画轮廓
			cv2.drawContours(self.img, good_contours, -1, (0, 255, 255), 5)

	# cv2.drawContours(self.img, contours, -1, (0, 255, 255), 5)
	# cv2.namedWindow("final_contours", 0)
	# cv2.imshow("final_contours", self.img)

	# 筛选箱子
	# 该方法暂时废弃了
	def filterlandmark(self):
		scoredict = {}
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		for a in self.landmarks:
			scoredict[str(a.id)] = 0
			for b in self.landmarks:
				if a == b:
					scoredict[str(a.id)] += 1
				elif abs(a.x - b.x) + abs(a.y - b.y) < 700:
					continue
				elif abs(b.x - 0.5 * cols) < 1000:
					continue
				elif abs(a.x - b.x) < 30 or abs(a.y - b.y) < 30:
					scoredict[str(a.id)] += 1
		good_box_indexs = [index for index, score in scoredict.items() if score > 2]
		good_boxes = []
		for box in self.landmarks:
			if str(box.id) in good_box_indexs:
				good_boxes.append(box)
		self.landmarks = good_boxes

	def find_nearest_bag(self):
		'''找到离钩子最近的袋子'''
		self.iner_computer_landmarks_location()  # 计算地标位置
		self.inner_computer_bags_location()  # 计算所有袋子定位
		if self.bags is None or len(self.bags) == 0:
			reply = QMessageBox.information(self,  # 使用infomation信息框
			                                "标题",
			                                "没有发现袋子，请核查实际情况！",
			                                QMessageBox.Yes | QMessageBox.No)
			print(reply)
			mylog_debug("没有发现袋子，请核查实际情况！")
			return None

		hockposition = self.inner_compute_hook_location()  # 计算钩子位置，为了移动钩子

		if hockposition is None:
			reply = QMessageBox.information(self,  # 使用infomation信息框
			                                "标题",
			                                "没有发现钩子，请核查实际情况！",
			                                QMessageBox.Yes | QMessageBox.No)
			print(reply)
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
		cv2.circle(self.img, nearest_bag.boxcenterpoint, 100, (0, 255, 0), thickness=3)
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
		if self.print_or_no:
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def inner_compute_hook_location(self):
		'''
		摄像头的位置只有X轴移动，在Y轴是固定的；激光灯理想的安装情况是，灯光的Y轴与钩子同步，跟随点击运动。
		已知量：灯光的位置、地标间隔实际距离、地标间隔图像距离，摄像头在Y轴上距离是固定的，不用去管
		:return:
		'''
		process = Preprocess(self.img)
		laster_binary, contour = process.processed_laster
		# cv2.imshow("binary1", laster_binary)
		if contour is None:
			return None

		cv2.drawContours(self.img, [contour], -1, (0, 255, 0), 1)  # 找到唯一的轮廓就退出即可
		laster = Laster(contour, laster_binary, id=0)
		laster.modify_box_content()
		self.laster = laster

		lasterposition = self.laster.boxcenterpoint  # 激光灯位置X轴较钩子小
		# cv2.putText(self.img, "lasterlocation->({},{})".format(lasterposition[0], lasterposition[1]),
		#             (lasterposition[0] + 100, lasterposition[1]),
		#             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
		# cv2.circle(self.img, lasterposition, 40, (0, 0, 255), -1)
		# TODO 实际钩子的距离需要根据灯光的距离计算出来
		hockposition = lasterposition  # 钩子的坐标
		# cv2.circle(self.img, hockposition, 60, (65, 105, 225), -1)
		cv2.putText(self.img, "hock location->({},{})".format(hockposition[0], hockposition[1]),
		            (hockposition[0] + 100, hockposition[1]),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (65, 105, 225), 2)

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

		# 两个点之间的实际距离
		land_mark_realspace = DISTANCE_SAMEXLANDMARK_SPACE if self.landmark_select_method == Landmark_Model_Select.CHOOSE_X_SAME else DISTANCE_SAMEYLANDMARK_SPACE
		real_distance = round(img_distance * land_mark_realspace / self.landmarkvirtualdistance, 2)

		x_move = round(point1_x - point2_x)
		real_x_distance = round(x_move * land_mark_realspace / self.landmarkvirtualdistance, 2)

		y_move = round(point1_y - point2_y)
		real_y_distance = round(y_move * land_mark_realspace / self.landmarkvirtualdistance, 2)
		return img_distance, real_distance, real_x_distance, real_y_distance

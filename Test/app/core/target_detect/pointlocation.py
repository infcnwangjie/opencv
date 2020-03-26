# 所有的功能点如下所示：
#
# 1、找到所有的袋子
#
# 2、找到所有的地标
#
# 3、移动钩子到所有袋子

# encoding:utf-8
import math
import random

import cv2
import numpy as np

from app.config import DISTANCE_LANDMARK_SPACE
from app.core.preprocess.preprocess import Preprocess
from app.core.target_detect.models import Box, LandMark, Bag
from app.core.target_detect.digitdetect import DigitDetector

BAG_AND_LANDMARK = 0
ONLY_BAG = 1
ONLY_LANDMARK = 2


class PointLocationService:
	def __init__(self, img, print_or_no=True):
		self.img = img
		self.bags = []
		self.landmarks = []
		self.processfinish = False
		self.print_or_no = print_or_no

	def __enter__(self):
		return self

	# 预处理_定位袋子
	def preprocess_for_bag(self, colorlow=(61, 83, 31), colorhigh=(81, 255, 250)):
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		binary = cv2.medianBlur(binary, 3)
		return binary

	# 计算袋的坐标
	def computer_bags_location(self, digitdetector):
		bag_binary = self.preprocess_for_bag(colorlow=[120, 50, 50], colorhigh=[180, 255, 255])
		contours, _hierarchy = cv2.findContours(bag_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours is None or len(contours) == 0:
			return
		# 大小适中的轮廓，过小的轮廓被去除了
		moderatesize_countours = []
		boxindex = 0
		for countour in contours:
			countour_rect = cv2.boundingRect(countour)
			rect_x, rect_y, rect_w, rect_h = countour_rect
			if cv2.contourArea(countour) > 1500 and rect_h < 100:
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
	def computer_landmarks_location(self, digitdetector):
		process = Preprocess(self.img)
		binary_image, contours = process.processedimg
		if contours is None or len(contours) == 0:
			return

		boxindex = 0
		for countour in contours:
			box = LandMark(countour, binary_image, id=boxindex, digitdetector=digitdetector)
			box.modify_box_content(digitdetector, no_num=True)
			boxindex += 1
			self.landmarks.append(box)
		# 过滤袋子->将面积较小的袋子移除
		self.filterbox()
		good_contours = []
		for box in self.landmarks:
			# 标记坐标和值
			cv2.putText(self.img, box.box_content, (box.boxcenterpoint[0] + 50, box.boxcenterpoint[1] + 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
			good_contours.append(box.contour)
		# 用黄色画轮廓
		cv2.drawContours(self.img, good_contours, -1, (0, 255, 255), 5)

	# 筛选箱子
	def filterbox(self):
		scoredict = {}
		for a in self.landmarks:
			scoredict[str(a.id)] = 0
			for b in self.landmarks:
				if a == b:
					scoredict[str(a.id)] += 1
				elif abs(a.x - b.x) + abs(a.y - b.y) < 700:
					continue
				elif abs(a.x - b.x) < 30 or abs(a.y - b.y) < 30:
					scoredict[str(a.id)] += 1
		good_box_indexs = [index for index, score in scoredict.items() if score > 2]
		good_boxes = []
		for box in self.landmarks:
			if str(box.id) in good_box_indexs:
				good_boxes.append(box)
		self.landmarks = good_boxes

	# 这一步定位所有摄像头看到的目标，并且计算出坐标
	def computelocations(self, flag=BAG_AND_LANDMARK):
		tool = DigitDetector()
		tool.practise()
		# 同时定位袋子和地标
		if flag == BAG_AND_LANDMARK:
			self.computer_bags_location(tool)
			self.computer_landmarks_location(tool)
		# 	仅仅定位袋子
		elif flag == ONLY_BAG:
			self.computer_bags_location(tool)
		# 	仅仅定位地标
		elif flag == ONLY_LANDMARK:
			self.computer_landmarks_location(tool)

	# 将坐标打印到图片上
	def print_location_onimg(self):
		# cv2.imshow('im', self.img)
		cv2.namedWindow('im', cv2.WINDOW_KEEPRATIO)
		cv2.imshow("im", self.img)
		return self.im

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.print_or_no:
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def compute_hook_location(self):
		'''
		摄像头的位置只有X轴移动，在Y轴是固定的；激光灯理想的安装情况是，灯光的Y轴与钩子同步，跟随点击运动。
		已知量：灯光的位置、地标间隔实际距离、地标间隔图像距离，摄像头在Y轴上距离是固定的，不用去管
		:return:
		'''
		lasterposition = (2337, 1702)  # 激光灯位置X轴较钩子小

		cv2.putText(self.img, "lasterlocation->({},{})".format(lasterposition[0], lasterposition[1]),
		            (lasterposition[0] + 100, lasterposition[1]),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
		cv2.circle(self.img, lasterposition, 40, (0, 0, 255), -1)
		hockposition = (2337, 1902)  # 钩子的坐标
		cv2.circle(self.img, hockposition, 60, (0, 255, 0), -1)
		cv2.putText(self.img, "hock location->({},{})".format(hockposition[0], hockposition[1]),
		            (hockposition[0] + 100, hockposition[1]),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

		return (2337, 1902)

	@property
	def landmark_virtual_distance(self):
		if not hasattr(self, 'landmarkvirtualdistance') or self.landmarkvirtualdistance is None:
			# 寻找左侧的地标
			left_landmarks, right_landmarks = [], []
			grayimg = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
			rows, cols = grayimg.shape
			for landmark in self.landmarks:
				if landmark.x < cols * 0.5:
					left_landmarks.append(landmark)
				else:
					right_landmarks.append(landmark)
			# 选择挨着的两个地标
			left_landmarks = sorted(left_landmarks, key=lambda position: position.y)
			landmark1, landmark2 = left_landmarks[0:2]
			self.landmarkvirtualdistance = abs(round(landmark1.y - landmark2.y, 3))
		return self.landmarkvirtualdistance

	def compute_distance(self, point1, point2):
		'''
		:param point1:
		:param point2:
		:return: 两点之间图像像素距离，两点之间真实距离，
		两点之间X轴需要真实移动距离，两点之间Y轴需要真实移动距离
		'''

		point1_x, point1_y = point1
		point2_x, point2_y = point2
		# 计算图像中两个点之间的像素距离
		img_distance = math.sqrt(
			math.pow(point2_y - point1_y, 2) + math.pow(point2_x - point1_x, 2))

		# 两个点之间的实际距离
		real_distance = round(img_distance * DISTANCE_LANDMARK_SPACE / self.landmark_virtual_distance, 2)

		x_move = round(point1_x - point2_x)
		real_x_distance = round(x_move * DISTANCE_LANDMARK_SPACE / self.landmark_virtual_distance, 2)

		y_move = round(point1_y - point2_y)
		real_y_distance = round(y_move * DISTANCE_LANDMARK_SPACE / self.landmark_virtual_distance, 2)
		return img_distance, real_distance, real_x_distance, real_y_distance

	def move(self):
		# 寻找钩子
		hockposition = self.compute_hook_location()

		distance_dict = {}
		for bag in self.bags:
			img_distance, real_distance, _movex, _movey = self.compute_distance(bag.boxcenterpoint, hockposition)
			distance_dict[str(int(img_distance))] = bag

		smallestindex = min(distance_dict.keys(), key=lambda item: int(item))
		nearest_bag = distance_dict[str(smallestindex)]

		img_distance, real_distance, move_x, move_y = self.compute_distance(nearest_bag.boxcenterpoint, hockposition)
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

		# 袋子质心到钩子的直线
		cv2.line(self.img, hockposition, nearest_bag.boxcenterpoint, (0, 255, 255), thickness=3)
		# 袋子质心与钩子所在x轴的垂直线
		cv2.line(self.img, nearest_bag.boxcenterpoint, (nearest_bag.boxcenterpoint[0], hockposition[1]), (0, 255, 255),
		         thickness=3)

		word_position_y = (int(bag.x - 60), hockposition[1] - 60)

		cv2.putText(self.img, moveinfo_y, word_position_y, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

		word_position_x = (int(bag.x + abs(0.5 * (bag.x - hockposition[0]))), hockposition[1] + 100)
		cv2.putText(self.img, moveinfo_x, word_position_x, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

		cv2.line(self.img, (nearest_bag.boxcenterpoint[0], hockposition[1]), hockposition, (0, 255, 255),
		         thickness=3)

		if self.print_or_no:
			self.print_location_onimg()
		return self.img

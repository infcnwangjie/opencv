# 所有的功能点如下所示：
#
# 1、找到所有的袋子
#
# 2、找到所有的地标
#
# 3、移动钩子到所有袋子

import cv2
import numpy as np

from app.core.preprocess.preprocess import Preprocess
from app.core.target_detect.models import Box
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
				box = Box(countour, bag_binary, id=boxindex)
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
		# cv2.namedWindow('landmark_binary', cv2.WINDOW_KEEPRATIO)
		# cv2.imshow("landmark_binary", binary_image)
		# contours, hierarchy = cv2.findContours(process.processedimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours is None or len(contours) == 0:
			return

		boxindex = 0
		for countour in contours:
			box = Box(countour, binary_image, id=boxindex, digitdetector=digitdetector)
			box.modify_box_content(digitdetector, no_num=False)
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
		# 用紫色画轮廓
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
	def location_objects(self, flag=BAG_AND_LANDMARK):
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
		if self.print_or_no:
			self.print_location_onimg()

	# 将坐标打印到图片上
	def print_location_onimg(self):
		# cv2.imshow('im', self.img)
		cv2.namedWindow('im', cv2.WINDOW_KEEPRATIO)
		cv2.imshow("im", self.img)

	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.print_or_no:
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def next_move(self):
		# assert len(self.bags)==0,"bag detect fail"
		for bag in self.bags:
			print(bag.box_content)

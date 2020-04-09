# -*- coding: utf-8 -*-
import cv2

from app.core.target_detect.digitdetect import DigitDetector


class DigitLocation:
	def __init__(self, digitvalue, locationpoint, bagcenterpoint, boxid):
		self.boxid = boxid
		self.digitvalue = digitvalue
		self.boxcenterpoint_x, self.boxcenterpoint_y = bagcenterpoint
		self.locationpoint_x, self.locationpoint_y = locationpoint

	def __str__(self):
		return "boxid:{boxid}-digitvalue:{digitvalue}-x:{x}-y:{y}-box_x:{box_x}-box_y:{box_y}".format(
			boxid=self.boxid,
			digitvalue=self.digitvalue,
			x=self.locationpoint_x,
			y=self.locationpoint_y,
			box_x=self.boxcenterpoint_x,
			box_y=self.boxcenterpoint_y)


# 目标体
class Box:
	'''
	Box(contour，grayimage,id,numdetector)
	'''

	def __init__(self, contour, img, id=1):
		if img is None:
			raise Exception("box img must not none")
		self.id, self.img = id, img
		self.contour = contour
		self.box = cv2.boundingRect(contour)
		self.x, self.y, self.w, self.h = self.box
		self.boxcenterpoint = (self.x + round(self.w * 0.5), self.y + round(self.h * 0.5))
		self.x = self.x + round(self.w * 0.5)
		self.y = self.y + round(self.h * 0.5)

		# self.compute_iner_contours()
		self.status = True

	# 修改目标物的显示内容
	def modify_box_content(self, digitdetector, no_num=True):
		# 如果box内部没有内部轮廓，就直接退出循环
		if no_num:
			self.box_content = "bag_location:" + "->(" + str(self.boxcenterpoint[0]) + "," + str(
				self.boxcenterpoint[1]) + ")"
			return

		for digital_contour in self.inercontours:
			[digit_point_x, digit_point_y, digit_contor_width, digit_contor_height] = cv2.boundingRect(
				digital_contour)
			roi = self.thresh[digit_point_y:digit_point_y + digit_contor_height,
			      digit_point_x:digit_point_x + digit_contor_width]
			results = digitdetector.readnum(roi)

			roi_digitvalue = str(int((results[0][0])))
			boxdigitlocation = DigitLocation(digitvalue=roi_digitvalue, boxid=self.id,
			                                 bagcenterpoint=self.boxcenterpoint,
			                                 locationpoint=(
				                                 self.x + digit_point_x, self.y + digit_point_y))
			self.digitLocations.append(boxdigitlocation)

		# cv2.drawContours(self.img, self.inercontours, -1, (0, 0, 128), 5)
		# cv2.drawContours(self.img, self.inercontours)
		# if self.digitLocations is None or len(self.digitLocations) == 0:
		#     return
		# self.digitLocations.sort(key=lambda location: location.locationpoint_x, reverse=False)
		# # 用于拼接数字，当然遇到6,8的时候回检测出两个轮廓，用x轴之差决定是否拼接
		# last_point_x, box_digitnum = 0, ""
		# for location in self.digitLocations:
		#     current_x = location.locationpoint_x
		#     if current_x - last_point_x > 10:
		#         box_digitnum += location.digitvalue
		#     last_point_x = current_x
		# self.box_content = box_digitnum + "->(" + str(self.boxcenterpoint[0]) + "," + str(
		#     self.boxcenterpoint[1]) + ")"


# 地标
class LandMark(Box):
	def __init__(self, contour, img, id=1, ori_img=None):
		super().__init__(contour, img, id=1)
		self.roi_contours, self.thresh = None, None
		self.box_content = ""
		self.has_compute_contours = False
		self.ori_img = ori_img
		# 数字检测对象
		self.digitdetector = DigitDetector()
		# ret, thresh = cv2.threshold(self.img, 30, 255, cv2.THRESH_BINARY_INV)
		# cv2.imshow("inv",thresh)
		# self.inv_img =thresh
		# 内部使用，计算box中的所有数字轮廓

	def compute_iner_contours(self):
		if self.roi_contours is None and self.has_compute_contours == False:
			self.has_compute_contours = True
			x, y, w, h = self.box
			# ret, thresh = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY)  # 简单阈值
			# cv2.imshow("ini", thresh)
			gray = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2GRAY)
			# cv2.imshow("gray",gray)
			roi_img = gray[y + 1:y + h, x + 1:x + w]


			if roi_img is None:
				raise Exception("roi_img 拷贝失败！")

			ret, thresh = cv2.threshold(roi_img, 40, 255, cv2.THRESH_BINARY)  # 简单阈值
			# cv2.namedWindow("roi", 0)
			# cv2.imshow("roi", thresh)
			# ret, thresh = cv2.threshold(roi_img, 30, 255, cv2.THRESH_BINARY)  # 简单阈值
			# 在特征区域中再次寻找轮廓
			roi_contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			print("内部ioi num is {}".format(len(roi_contours)))
			roi_contours = sorted(roi_contours, key=lambda c: cv2.contourArea(c), reverse=True)
			# cv2.drawContours(thresh, roi_contours, -1, (255, 255, 0), 5)

			self.roi_contours = roi_contours[0:2]
			self.thresh = thresh
		# 内部使用,通过轮廓面积过滤轮廓

	def __contours_area_filter(self, c, minarea=100, maxarea=3000):
		[x1, y1, w1, h1] = cv2.boundingRect(c)
		area = cv2.contourArea(c)
		# and h1 > h * 0.50
		return maxarea > area > minarea

	@property
	def inercontours(self):
		if self.has_compute_contours:
			return self.roi_contours
		else:
			self.compute_iner_contours()
		return self.roi_contours

	# 修改目标物的显示内容
	def modify_box_content(self, no_num=True):
		# 如果box内部没有内部轮廓，就直接退出循环
		if no_num:
			self.box_content = "landmark:" + "->(" + str(self.boxcenterpoint[0]) + "," + str(
				self.boxcenterpoint[1]) + ")"
			return

		# cv2.namedWindow("binary_image", 0)
		# cv2.imshow("binary_image", self.img)

		digital_nums = []
		print("*" * 100)
		for digital_contour in self.inercontours:
			[digit_point_x, digit_point_y, digit_contor_width, digit_contor_height] = cv2.boundingRect(
				digital_contour)
			roi = self.thresh[digit_point_y:digit_point_y + digit_contor_height,
			      digit_point_x:digit_point_x + digit_contor_width]

			# cv2.namedWindow("roi", 0)
			# cv2.imshow("roi", roi)

			results = self.digitdetector.readnum(roi)
			# print(results)

			roi_digitvalue = str(int((results[0][0])))
			boxdigitlocation = DigitLocation(digitvalue=roi_digitvalue, boxid=self.id,
			                                 bagcenterpoint=self.boxcenterpoint,
			                                 locationpoint=(
				                                 self.x + digit_point_x, self.y + digit_point_y))
			digital_nums.append(boxdigitlocation)
			#
		# cv2.imshow("thresh", self.thresh)
		# cv2.drawContours(self.img, self.inercontours, -1, (0, 0, 255), 2)
		# cv2.drawContours(self.img, self.inercontours)
		if digital_nums is None or len(digital_nums) == 0:
			return
		digital_nums.sort(key=lambda location: location.locationpoint_x, reverse=False)
		# 用于拼接数字，当然遇到6,8的时候回检测出两个轮廓，用x轴之差决定是否拼接
		last_point_x, box_digitnum = 0, ""
		for location in digital_nums:
			current_x = location.locationpoint_x
			if current_x - last_point_x > 10:
				box_digitnum += location.digitvalue
			last_point_x = current_x
		self.box_content = box_digitnum + "->(" + str(self.boxcenterpoint[0]) + "," + str(
			self.boxcenterpoint[1]) + ")"



# 袋子
class Bag(Box):
	def modify_box_content(self, digitdetector, no_num=True):
		# 如果box内部没有内部轮廓，就直接退出循环
		self.box_content = "bag:" + "(" + str(self.boxcenterpoint[0]) + "," + str(
			self.boxcenterpoint[1]) + ")"


# 激光灯
class Laster(Box):
	def modify_box_content(self):
		# 如果box内部没有内部轮廓，就直接退出循环
		self.box_content = "laster:" + "(" + str(self.boxcenterpoint[0]) + "," + str(
			self.boxcenterpoint[1]) + ")"


# 钩子
class Hock(Box):
	def modify_box_content(self):
		# 如果box内部没有内部轮廓，就直接退出循环
		self.box_content = "hock:" + "->(" + str(self.boxcenterpoint[0]) + "," + str(
			self.boxcenterpoint[1]) + ")"

# -*- coding: utf-8 -*-
import cv2


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
		self.status = True

	# 修改目标物的显示内容
	def modify_box_content(self, no_num=True):
		# 如果box内部没有内部轮廓，就直接退出循环
		if no_num:
			self.box_content = "bag_location:" + "->(" + str(self.boxcenterpoint[0]) + "," + str(
				self.boxcenterpoint[1]) + ")"
			return


# 袋子
class Bag(Box):
	def modify_box_content(self, no_num=True):
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

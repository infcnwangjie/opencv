# -*- coding: utf-8 -*-
import re
from collections import defaultdict

import cv2

from app.config import IMG_HEIGHT, IMG_WIDTH

rows, cols = IMG_HEIGHT, IMG_WIDTH

SLIDE_WIDTH, SLIDE_HEIGHT = 25, 25

FOND_RECT_WIDTH, FOND_RECT_HEIGHT = 70, 70

LEFT_START, LEFT_END = 150, 175

RIGHT_START, RIGHT_END = 766, 796


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
	def __init__(self, contour, img, id=1):
		super().__init__(contour, img, id)
		self.finish_move = False

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


class NearLandMark:
	# __slots__ = ['col', 'row', '_slide_img', '_similarity', '_roi', 'direct']

	def __init__(self, col, row, slide_img, similarity=0):
		self.col = col
		self.row = row
		self._slide_img = slide_img
		self._similarity = similarity
		self._roi = None
		self.direct = 'left' if self.col < 0.5 * cols else 'right'  # 0 :L 1:R
		self.landmark_name = None  # 在被识别的时候被赋值


	@property
	def positioninfo(self):
		return self.col, self.row, self._slide_img

	@property
	def similarity(self):
		return self._similarity

	@similarity.setter
	def similarity(self, value):
		self._similarity = value

	def __str__(self):
		return "{}:({},{})".format(self.landmark_name,self.col,self.row)


# @property
# def roi(self):
# 	return self._roi
#
# @roi.setter
# def roi(self, value):
# 	value.times += 1
# 	self._roi = value


class TargetRect:
	__slots__ = ['point1', 'point2']

	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	def slider_in_rect(self, slide_obj: NearLandMark = None, slide_col=None, slide_row=None):
		if slide_obj:
			slide_col, slide_row, _img = slide_obj.data
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		else:
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		if self.point1[0] < slide_point1[0] and self.point1[1] < slide_point1[1] and self.point2[0] > slide_point2[
			0] and self.point2[1] > slide_point2[1]:
			return True
		return False


class LandMarkRoi:
	def __init__(self, img, label, id=None):
		self.roi = img
		self.id = id
		self.label = label
		self._times = 0
		self.landmark = None

	def add_slide_window(self, slide_window: NearLandMark):
		# with self.lock:
		if self.landmark is None:
			self.landmark = slide_window
		else:
			col, row, similar = self.landmark.col, self.landmark.row, self.landmark.similarity
			col1, row1, similar1 = slide_window.col, slide_window.row, slide_window.similarity
			if similar < similar1:
				self.landmark = slide_window

	@property
	def times(self):
		return self._times

	@times.setter
	def times(self, value):
		self._times = value
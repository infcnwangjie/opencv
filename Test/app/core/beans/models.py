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
		# if img is None:
		# 	raise Exception("box img must not none")
		self.id, self.img = id, img
		self.contour = contour
		self.box = cv2.boundingRect(contour)
		self.x, self.y, self.w, self.h = self.box
		self.boxcenterpoint = (self.x + round(self.w * 0.5), self.y + round(self.h * 0.5))
		self.area = (self.x - 50, self.y - 50, self.x + 50, self.y + 50)
		# self.x = self.x + round(self.w * 0.5)
		# self.y = self.y + round(self.h * 0.5)
		self.status = True

	@property
	def width(self):
		return self.w

	@property
	def height(self):
		return self.h

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
		self.status_map = dict(move_close=True, drop_hock=False, pull_hock=False, put_down_bag=False)



	def modify_box_content(self, no_num=True):
		self.box_content = "(" + str(self.x) + "," + str(
			self.y) + ")"

	def __str__(self):
		return "( x:" + str(self.boxcenterpoint[0]) + ",y:" + str(
			self.boxcenterpoint[1]) + ",p:" + str(self.width) + ")"


# 激光灯
class Laster(Box):
	def modify_box_content(self):
		# 如果box内部没有内部轮廓，就直接退出循环
		self.box_content = "laster:" + "(" + str(self.x) + "," + str(
			self.y) + ")"


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
		self.landmark_name = None  # 当识别是不是某个地标ROI的时候调用
		self.maybe_labels = []
		self.width = 0
		self.height = 0

	def add_maybe_label(self, label):
		'''颜色相近的地标有时候仅仅根据近似程度无法区分，
		这时候对于某地标区域来说无法界定所属标签'''
		self.maybe_labels.append(label)

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
		return "{}:({},{})".format(self.landmark_name, self.col, self.row)


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


class BagRoi:
	def __init__(self, img, id=1):
		self.roi = img
		self.x, self.y = 0, 0
		self.id = id

	def set_position(self, x, y):
		self.x, self.y = x, y

	def get_position(self):
		return self.x, self.y


class SupportRefRoi:
	def __init__(self, img, id=1):
		self.roi = img
		self.x, self.y = 0, 0
		self.id = id

	def set_position(self, x, y):
		self.x, self.y = x, y

	def get_position(self):
		return self.x, self.y


class LandMarkRoi:
	def __init__(self, img, label, id=None):
		self.roi = img
		self.id = id
		self.label = label
		self._times = 0
		self.landmark = None
		self._row_no = None  # label中的数字部分
		self.possibles = []

	def add_to_possibel(self, slide_window: NearLandMark):
		self.possibles.append(slide_window)

	def set_match_obj(self, slide_window: NearLandMark):
		# with self.lock:
		# if self.landmark is None:
		# 	self.landmark = slide_window
		# else:
		# 	col, row, similar = self.landmark.col, self.landmark.row, self.landmark.similarity
		# 	col1, row1, similar1 = slide_window.col, slide_window.row, slide_window.similarity
		# 	if similar <= similar1:
		# 		self.landmark = slide_window
		self.landmark = slide_window

	@property
	def times(self):
		return self._times

	@times.setter
	def times(self, value):
		self._times = value

	@property
	def no(self):
		result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', self.label)
		if result is not None:
			return int(result.group(1))
		else:
			return -1

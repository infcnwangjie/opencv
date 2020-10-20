# -*- coding: utf-8 -*-
import re
from collections import defaultdict
import queue
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

	@classmethod
	def calc_landmarkcenter(cls, c):
		# area = cv2.contourArea(c)
		rect = cv2.boundingRect(c)
		x, y, w, h = rect
		M = cv2.moments(c)
		try:
			center_x = int(M["m10"] / M["m00"])
			center_y = int(M["m01"] / M["m00"])
		except:
			center_x, center_y = x, y
		return center_x, center_y

	def __init__(self, contour, id=None):
		# if img is None:
		# 	raise Exception("box img must not none")
		self.id = id
		# 用于解决地标检测不到的时候，使用的坐标
		self.img_x, self.img_y = None, None
		self.contour = contour
		self.box = cv2.boundingRect(contour)
		self.x, self.y, self.w, self.h = self.box
		# self.boxcenterpoint = (self.x + round(self.w * 0.5), self.y + round(self.h * 0.5))

		# self.boxcenterpoint = (self.x - round(self.w * 0.5), self.y - round(self.h * 0.5))
		self.center_x, self.center_y = self.calc_landmarkcenter(contour)

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
		pass


class BagContour:
	def __init__(self, c):
		self.__c = c
		self.__row_sort_index = 0
		self.__col_sort_index = 0

	@property
	def c(self):
		return self.__c

	@c.setter
	def contour(self, value):
		self.__c = value

	@property
	def row_sort_index(self):
		return self.__row_sort_index

	@row_sort_index.setter
	def row_sort_index(self, row_sort_index):
		self.__row_sort_index = row_sort_index

	@property
	def col_sort_index(self):
		return self.__col_sort_index

	@col_sort_index.setter
	def col_sort_index(self, col_sort_index):
		self.__col_sort_index = col_sort_index

	def __str__(self):
		return "x_index:{},y_index:{}".format(self.col_sort_index, self.row_sort_index)


class BagCluster:
	def __init__(self, cent_x, cent_y):
		self.__center_x = cent_x
		self.__center_y = cent_y
		self.__row_sort_index = 0
		self.__col_sort_index = 0

	@property
	def cent_x(self):
		return self.__center_x

	@cent_x.setter
	def cent_x(self, value):
		self.__center_x = value

	@property
	def cent_y(self):
		return self.__center_y

	@cent_y.setter
	def cent_y(self, value):
		self.__center_y = value

	@property
	def row_sort_index(self):
		return self.__row_sort_index

	@row_sort_index.setter
	def row_sort_index(self, row_sort_index):
		self.__row_sort_index = row_sort_index

	@property
	def col_sort_index(self):
		return self.__col_sort_index

	@col_sort_index.setter
	def col_sort_index(self, col_sort_index):
		self.__col_sort_index = col_sort_index

	def __str__(self):
		return "x_index:{},y_index:{}".format(self.col_sort_index, self.row_sort_index)


# 袋子
class Bag:
	'''
			choose: 选为目标袋子
			move_close: 钩子移动靠近目标袋子
			drop_hock:  放下钩子
			pull_hock:  拉起钩子
			put_down_bag:   放下袋子到传送带
			finish: 完成运输
	'''

	def __init__(self, centx=None, centy=None, id=None):
		'''
		centx 质心x,centy 质心y
		x ，y 要废弃
		:param bagcontour:
		:param centx:
		:param centy:
		'''
		self.id = id
		self.status_map = dict(choose=False, move_close=False, drop_hock=False, hock_suck=False, pull_hock=False,
		                       put_down_bag=False,
		                       finish_move=False)

		if centx is not None and centy is not None:
			self.y, self.cent_y = centy, centy
			self.cent_x, self.x = centx, centx
		else:
			self.x, self.cent_x, self.y, self.cent_y = 0, 0, 0, 0

		self.height = None
		self.step = None
		self.step_pointer = -1
		self.down_hock_much = 0
		self.if_suck = False
		self.suck_times = 0
		# self.suckhock_positions = {}
		self.previous_position = None  # 过去的坐标值
		self.positionx_list = []  # 袋子坐标X值每帧的值
		self.positiony_list = []  # 袋子坐标Y值每帧的值

	# self.position_queue.put()

	def modify_box_content(self):
		self.box_content = "bag:(" + str(self.cent_x) + "," + str(
			self.cent_y) + "," + str(self.height) + ")"

	def __str__(self):
		return "( x:" + str(self.cent_x) + ",y:" + str(
			self.cent_y) + ",p:" + str(self.width) + ")"


# 激光灯
class Laster(Box):
	def modify_box_content(self):
		# 如果box内部没有内部轮廓，就直接退出循环
		self.box_content = "laster:" + "(" + str(self.x) + "," + str(
			self.y) + ")"


# 钩子
class Hock(Box):
	def __init__(self, contour=None):
		super().__init__(contour, id=None)

	def set_position(self,cent_x,cent_y):
		self.center_x=cent_x
		self.center_y=cent_y

	def modify_box_content(self):
		# 如果box内部没有内部轮廓，就直接退出循环
		if not hasattr(self, 'box_content'):
			self.box_content = None
		self.box_content = "light:" + "(" + str(self.center_x) + "," + str(
			self.center_y) + ")"


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


class HockRoi:
	def __init__(self, img):
		self.img = img
		self.x, self.y = 0, 0

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
	all_roi_instances = []  # 已经认证过的roi

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

	def set_match_obj(self, slide_window: NearLandMark, target):
		# self.all_roi_instances.append(self)
		self.landmark = slide_window

	# def set_match_obj(self, slide_window: NearLandMark, target):
	# 	if len(self.all_roi_instances) == 0:
	# 		self.landmark = slide_window
	# 		self.all_roi_instances.append(self)
	# 	else:
	# 		for roi in self.all_roi_instances:
	# 			# 解决NO3_L 跑到NO1_L 的位置
	# 			if roi.no != self.no and roi.direct == self.direct and abs(roi.landmark.row - slide_window.row) < 50:
	# 				# 如果先找到的NO1_L,NO3_L后找到，那么roi里面已经有了NO1_L
	# 				if roi.no < self.no:
	# 					same_rows = list(
	# 						filter(lambda x: x.no == self.no and self.direct != x.direct, self.all_roi_instances))
	# 					if same_rows is not None and len(same_rows) > 0:
	# 						same_row = same_rows[0]
	#
	# 						x, y, w, h = roi.landmark.col, same_row.landmark.row, same_row.landmark.width, same_row.landmark.height
	#
	# 						landmark_obj = NearLandMark(x, y,
	# 						                            target[y:y + h, x:x + w])
	# 						landmark_obj.add_maybe_label(self.label)
	#
	# 						self.landmark = landmark_obj
	# 						self.all_roi_instances.append(self)
	#
	# 				# 如果先找到了NO3_L,NO1_L 后找到，那么roi里面已经有了NO3_L
	# 				else:
	# 					# del roi
	# 					same_rows = list(
	# 						filter(lambda x: x.no == roi.no and roi.direct != x.direct, self.all_roi_instances))
	# 					if same_rows is not None and len(same_rows) > 0:
	# 						same_row = same_rows[0]
	#
	# 						x, y, w, h = slide_window.col, same_row.landmark.row, same_row.landmark.width, same_row.landmark.height
	# 						landmark_obj = NearLandMark(x, y,
	# 						                            target[y:y + h, x:x + w])
	# 						landmark_obj.add_maybe_label(roi.label)
	# 						roi.landmark = landmark_obj
	# 					# 添加NO1_L
	# 					self.landmark = slide_window
	# 					self.all_roi_instances.append(self)
	#
	#
	#
	# 			# 解决NO3_L 与NO3_R Y轴相差过大
	# 			if self.label != roi.label and self.no == roi.no and abs(roi.landmark.row - slide_window.row) > 60:
	# 				break
	#
	# 			# 解决NO1_L 与 NO3_L X轴相差过大
	# 			if self.label != roi.label and self.direct == roi.direct and abs(
	# 					roi.landmark.col - slide_window.col) > 50:
	# 				break
	# 		else:
	# 			self.landmark = slide_window
	# 			self.all_roi_instances.append(self)

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

	@property
	def direct(self):
		result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', self.label)
		if result is not None:
			return result.group('direct')


class BagLocation:
	def __init__(self, cent_x, cent_y, finish=False, id=0):
		self.cent_x = cent_x
		self.cent_y = cent_y
		self.temp_points = [(cent_x, cent_y)]
		self.finish = finish
		self.id = id

	def count(self):
		return len(self.temp_points)

	def check_same(self, x, y, id=0):
		dif_x = abs(x - self.cent_x)
		dif_y = abs(y - self.cent_y)
		return (dif_x < 100 and dif_y < 100) or (id == self.id)

	def __average_point(self):
		import numpy
		x_avg = numpy.average([x for x, y in self.temp_points])
		y_avg = numpy.average([y for x, y in self.temp_points])
		return int(x_avg), int(y_avg)

	def add_point(self, bag):
		dif_x = abs(bag.cent_x - self.cent_x)
		dif_y = abs(bag.cent_y - self.cent_y)

		# if dif_x < 5 and dif_y < 5:
		# 	return

		if dif_x < 50 or dif_y < 50:
			self.temp_points.append((bag.cent_x, bag.cent_y))
			self.cent_x, self.cent_y = self.__average_point()

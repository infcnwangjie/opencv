# -*- coding: utf-8 -*-
import math
import os
import time
from collections import defaultdict
from itertools import chain
from time import sleep
import numpy as np
import cv2
import PyQt5.Qt
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMessageBox

# import app.core.autowork.detector.BaseDetector
# from app.core.autowork.detector import LandMarkDetecotr, BaseDetector

from app.config import *
from app.core.autowork.detector import *
# from app.core.autowork.detector import LandMarkDetecotr
from app.core.beans.models import Bag
from collections import defaultdict

from app.core.support.shapedetect import ShapeDetector, Shape
from app.log.logtool import logger


# ------------------------------------------------
# 名称：DetectorHandle
# 作者：王杰  2020-4-15
# ------------------------------------------------
class DetectorHandle(QObject):
	instance = None
	current_target_x, current_target_y = 0, 0  # 当前目标袋子坐标 x,y
	current_hock_x, current_hock_y = 0, 0  # 当前钩子坐标 x,y
	error_x, error_y = 0, 0  # 当前x坐标纠偏，当前y坐标纠偏
	input_move_instructs = []  # 所有的移动指令
	hock_points = []
	bags_info = {}  # key:sortx_index_1 join sorty_index_2,value:[x,y]
	error_info = {}  # 异常信息字典
	temp_bag_positions = []
	bag_location_progress = defaultdict(bool)
	send_warn_info = pyqtSignal(str)
	keep_y_move = True
	keep_x_move = True
	laster_status = True
	slow = False
	x_distance = {'s': 0}
	y_distance = {'e': 0}
	retry_find_landmark = 0

	def __new__(cls, *args, **kwargs):
		if cls.instance is None:
			cls.instance = super().__new__(cls)
		return cls.instance

	def __init__(self, plchandle):
		super().__init__()
		self.bags = []
		self.current_bag = None
		self.hock = None
		self.last_hock = None
		self.laster = None
		self.last_laster = None
		self.finish_job = False
		self._misshock_times = 0
		self.landmark_detect = LandMarkDetecotr()
		self.bag_detect = BagDetector()
		self.laster_detect = LasterDetector()
		self.hock_detect = HockDetector()
		self.history_bags = []
		self.history_laster_travel = []
		self.plchandle = plchandle
		self.status_show = None  # 状态栏更新消息
		self._modify_bagposition_show = [0, 0]
		self._modify_hockposition_show = [0, 0]

	@property
	def modify_bagposition(self):
		return self._modify_bagposition_show

	@property
	def modify_hockposition(self):
		return self._modify_hockposition_show

	@modify_bagposition.setter
	def modify_bagposition(self, position):
		self._modify_bagposition_show = position

	@modify_hockposition.setter
	def modify_hockposition(self, position):
		self._modify_hockposition_show = position

	# ------------------------------------------------
	# 名称：plc_connect
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def plc_connect(self):
		plcconnect = self.plchandle.is_open()
		return plcconnect

	# ------------------------------------------------
	# plc_check_errors
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def plc_check_errors(self):
		error = self.plchandle.check_error()
		return error

	# ------------------------------------------------
	# 名称：is_in_center
	# 作者：王杰  2020-7-24修改
	# ------------------------------------------------
	def is_in_center(self, show, bag_id):
		gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		bags, foreground = self.bag_detect.location_bags_withoutlandmark(original_img=show)

		# total_rows,_ignore_cols=self.bag_detect.shape
		# cv2.imshow("foreground",foreground)
		result = False
		for bag in bags:
			if bag.id == bag_id:
				if rows * 0.5 - 100 < bag.cent_y < rows * 0.5 + 100:
					result = True
				else:
					result = False
				break

		return result

	# ------------------------------------------------
	# 名称：scan_bags
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	# @cost_time
	def scan_bags(self, show):

		bag_positions = []
		plc_connect = self.plc_connect()

		if plc_connect == True or DEBUG == True:
			self.error_info = self.plc_check_errors()
		# print(self.error_info)

		if plc_connect == False and not DEBUG:
			logger("PLC连接失败", level='info')
			cv2.putText(show, "plc connect fail", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			return show, bag_positions, self.error_info

		perspective_img, find_landmark = self.landmark_detect.position_landmark(show)

		if find_landmark == True:
			perspective_copy_img = perspective_img.copy()
			# self.update_hockposition(perspective_img, original_img=None, find_landmark=find_landmark)

			bags, bag_forground = self.bag_detect.location_bags_withlandmark(perspective_img, perspective_copy_img,
			                                                                 find_landmark,
			                                                                 hock=self.hock, plchandle=self.plchandle)

			if bags is None or len(bags) == 0:
				return perspective_img, bag_positions, self.error_info

			if len(self.temp_bag_positions) == 0:
				for bag in bags:
					self.temp_bag_positions.append(BagLocation(bag.cent_x, bag.cent_y, False, bag.id))
			# self.bags_info[str(bag.id)] = [bag.cent_x, bag.cent_y]
			else:
				for bag in bags:
					# bag_finish_location = (self.hock is not None and abs(
					# 	bag.cent_y - self.hock.y) < 200)  # or self.is_in_center(show, bag.id) == True

					choosed_baglocations = list(
						filter(lambda points: points.check_same(bag.cent_x, bag.cent_y, bag.id),
						       self.temp_bag_positions))
					if len(choosed_baglocations) == 0:
						bag_location = BagLocation(bag.cent_x, bag.cent_y, False, bag.id)
						self.temp_bag_positions.append(bag_location)
					# self.bags_info[str(bag.id)] = [bag.cent_x, bag.cent_y]
					elif len(choosed_baglocations) == 1:
						bag_location = choosed_baglocations[0]
						if bag_location.finish == True:
							continue
						else:
							bag_location.add_point(bag)
							# self.bags_info[str(bag.id)] = [bag_location.cent_x, bag_location.cent_y]
							bag_location.finish = True
					else:
						continue

			bag_positions = [[bag_locaiton_item.cent_x, bag_locaiton_item.cent_y] for bag_locaiton_item in
			                 self.temp_bag_positions]  # if bag_locaiton_item.count() > 1
		else:
			self.hock_detect.has_stable = False
		# print(self.bags_info)
		logger("定位到{}个袋子".format(len(bag_positions)), level='info')
		return perspective_img, bag_positions, self.error_info

	def move_to_targetbag(self, show, bag_position, move_to_bag_x=True, move_to_bag_y=True):
		logger("向坐标({},{})移动".format(bag_position[0], bag_position[1]), 'info')
		temp_hock_position = []
		plc_connect = self.plc_connect()

		# plc连接失败就返回
		if DEBUG == False and plc_connect == False:
			logger("PLC连接失败", level='info')
			cv2.putText(show, "plc connect fail", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			return show, temp_hock_position, move_to_bag_x, move_to_bag_y, self.error_info
		else:
			if plc_connect == True:
				self.error_info = self.plc_check_errors()

		perspective_img, find_landmark = self.landmark_detect.position_landmark(show)
		# hock_x, hock_y = 0 if self.hock is None else self.hock.x, 0 if self.hock is None else self.hock.y
		correct_bag_x, correct_bag_y = int(bag_position[0]), int(bag_position[1])

		if find_landmark == True:
			return self.move_close_bag_withlandmark(bag_position, correct_bag_x, correct_bag_y, find_landmark,
			                                        perspective_img, temp_hock_position, move_to_bag_x, move_to_bag_y)
		# return self.move_close_bag_withlandmark_positionwhenstop(bag_position, correct_bag_x, correct_bag_y,
		#                                                          find_landmark,
		#                                                          perspective_img, temp_hock_position, move_to_bag_x,
		#                                                          move_to_bag_y)

		else:
			image_result = self.move_close_bag_withoutlandmark(correct_bag_x, correct_bag_y, show)
			self.hock_detect.has_stable = False
			return image_result, temp_hock_position, self.keep_x_move, self.keep_y_move, self.error_info

	def move_when_misslandmark(self):
		processed_x = []
		if len(self.x_distance.values()) > 0:
			for key, value in self.x_distance.items():
				processed_x.append(key)
				if key == 's':
					self.plchandle.move(east=0, west=0, south=value, nourth=0, up=0, down=0)
				else:
					self.plchandle.move(east=0, west=0, south=0, nourth=value, up=0, down=0)
		for delete_key in processed_x:
			del self.x_distance[delete_key]
		processed_y = []
		if len(self.y_distance.values()) > 0:
			for key, value in self.y_distance.items():
				processed_y.append(key)
				if key == 'w':
					self.plchandle.move(east=0, west=value, south=0, nourth=0, up=0, down=0)
				else:
					self.plchandle.move(east=value, west=0, south=0, nourth=0, up=0, down=0)
		for delete_key in processed_y:
			del self.y_distance[delete_key]

		self.plchandle.move_status(0)

	# ------------------------------------------------
	# 名称：move_close_bag_withlandmark
	# 作者：王杰  2020-8-xx
	# ------------------------------------------------
	def move_close_bag_withlandmark(self, bag_position, correct_bag_x, correct_bag_y, find_landmark,
	                                perspective_img, temp_hock_position, move_to_bag_x, move_to_bag_y):
		global move_tobag_retry_times
		self.keep_x_move, self.keep_y_move = move_to_bag_x, move_to_bag_y

		self.update_hockposition(perspective_img, original_img=None, find_landmark=find_landmark)

		if self.hock is None or not hasattr(self.hock, 'x') or self.hock.x is None \
				or self.hock.y is None or not hasattr(self.hock, 'y'):
			self.hock_detect.has_stable = False
			return perspective_img, temp_hock_position, self.keep_x_move, self.keep_y_move, self.error_info
		else:
			self.retry_find_landmark = 0
			hock_x, hock_y = self.hock.center_x - hock_x_error, self.hock.center_y - hock_y_error
			temp_hock_position = [hock_x, hock_y]
			self.modify_hockposition = [hock_x, hock_y]

			perspective_copy_img = perspective_img.copy()
			image_result = perspective_img

			correct_bag_x, correct_bag_y = self.modify_bag_position(correct_bag_x, correct_bag_y, find_landmark,
			                                                        perspective_copy_img, perspective_img)
			self.modify_bagposition = [correct_bag_x, correct_bag_y]

			logger("当前钩子坐标为({},{})，袋子坐标为({},{})".format(hock_x, hock_y, correct_bag_x, correct_bag_y), 'info')

			abs_x_distance, abs_y_distance, x_distance, y_distance = self.calculate_error(correct_bag_x, correct_bag_y,
			                                                                              hock_x, hock_y)

			self.x_distance = {'s' if x_distance > 0 else 'n': abs_x_distance}
			self.y_distance = {'w' if y_distance > 0 else 'e': abs_y_distance}

			if x_distance <= 50 and y_distance < 50: self.slow = True

			east, west, south, north, up, down = 0, 0, 0, 0, 0, 0

			if move_to_bag_x == True:
				self.move_x_direct(abs_x_distance, correct_bag_x, correct_bag_y, down, east, hock_x, hock_y,
				                   perspective_img, up, west, x_distance)

			if move_to_bag_y == True:
				self.move_y_direct(abs_y_distance, correct_bag_x, correct_bag_y, down, hock_x, hock_y, north,
				                   perspective_img, south, up, y_distance)

			if move_to_bag_y == False and move_to_bag_x == False:
				logger("钩子袋子已重叠", 'info')
				cv2.putText(perspective_img, "finish".format(hock_x, hock_y), (400, 500),
				            cv2.FONT_HERSHEY_SIMPLEX,
				            1.2,
				            (255, 255, 255), 2)
				self.keep_y_move, self.keep_x_move = False, False

			return image_result, temp_hock_position, self.keep_x_move, self.keep_y_move, self.error_info

	def move_close_bag_withlandmark_positionwhenstop(self, bag_position, correct_bag_x, correct_bag_y, find_landmark,
	                                                 perspective_img, temp_hock_position, move_to_bag_x, move_to_bag_y):
		global move_tobag_retry_times
		self.keep_x_move, self.keep_y_move = move_to_bag_x, move_to_bag_y

		move_status = self.plchandle.read_status()

		if move_status == 0 or move_status == "0":
			sleep(4)
			self.update_hockposition(perspective_img, original_img=None, find_landmark=find_landmark)
			if self.hock is None or not hasattr(self.hock, 'x') or self.hock.x is None \
					or self.hock.y is None or not hasattr(self.hock, 'y'):
				self.hock_detect.has_stable = False
				return perspective_img, temp_hock_position, self.keep_x_move, self.keep_y_move, self.error_info
			else:

				hock_x, hock_y = self.hock.center_x - hock_x_error, self.hock.center_y - hock_y_error
				temp_hock_position = [hock_x, hock_y]
				self.modify_hockposition = [hock_x, hock_y]

				perspective_copy_img = perspective_img.copy()
				image_result = perspective_img

				correct_bag_x, correct_bag_y = self.modify_bag_position(correct_bag_x, correct_bag_y, find_landmark,
				                                                        perspective_copy_img, perspective_img)
				self.modify_bagposition = [correct_bag_x, correct_bag_y]

				logger("当前钩子坐标为({},{})，袋子坐标为({},{})".format(hock_x, hock_y, correct_bag_x, correct_bag_y), 'info')

				abs_x_distance, abs_y_distance, x_distance, y_distance = self.calculate_error(correct_bag_x,
				                                                                              correct_bag_y,
				                                                                              hock_x, hock_y)

				self.x_distance = {'s' if x_distance > 0 else 'n': abs_x_distance}
				self.y_distance = {'w' if y_distance > 0 else 'e': abs_y_distance}

				if x_distance <= 50 and y_distance < 50: self.slow = True

				east, west, south, north, up, down = 0, 0, 0, 0, 0, 0

				if move_to_bag_x == True:
					self.move_x_direct(abs_x_distance, correct_bag_x, correct_bag_y, down, east, hock_x, hock_y,
					                   perspective_img, up, west, x_distance)

				if move_to_bag_y == True:
					self.move_y_direct(abs_y_distance, correct_bag_x, correct_bag_y, down, hock_x, hock_y, north,
					                   perspective_img, south, up, y_distance)

				if move_to_bag_y == False and move_to_bag_x == False:
					logger("钩子袋子已重叠", 'info')
					cv2.putText(perspective_img, "finish".format(hock_x, hock_y), (400, 500),
					            cv2.FONT_HERSHEY_SIMPLEX,
					            1.2,
					            (255, 255, 255), 2)
					self.keep_y_move, self.keep_x_move = False, False

				self.retry_find_landmark = 0

				return image_result, temp_hock_position, self.keep_x_move, self.keep_y_move, self.error_info

	def move_x_direct(self, abs_x_distance, correct_bag_x, correct_bag_y, down, east, hock_x, hock_y, perspective_img,
	                  up, west, x_distance):

		if direct_choice == 'y':
			self.keep_x_move = False
			self.plchandle.move_status(0)
		else:
			self.x_distance.pop('s' if x_distance > 0 else 'n')

			if abs_x_distance > permissible_distance:

				south = abs_x_distance if x_distance > 0 else 0
				north = abs_x_distance if x_distance < 0 else 0

				self.write_move_instruct(correct_bag_x, correct_bag_y, down, east, hock_x, hock_y, north,
				                         perspective_img, south, up, west)
				self.plchandle.move_status(0)

			else:
				self.keep_x_move = False  # abs_x_distance > permissible_distance
				self.plchandle.move_status(0)

	def calculate_error(self, correct_bag_x, correct_bag_y, hock_x, hock_y):
		y_distance, x_distance = correct_bag_y - hock_y, correct_bag_x - hock_x
		abs_y_distance = abs(y_distance)
		abs_x_distance = abs(x_distance)
		return abs_x_distance, abs_y_distance, x_distance, y_distance

	# 修正
	def modify_bag_position(self, correct_bag_x, correct_bag_y, find_landmark, perspective_copy_img, perspective_img):
		bags, bag_forground = self.bag_detect.location_bags_withlandmark(perspective_img, perspective_copy_img,
		                                                                 find_landmark
		                                                                 )

		if bags is None or len(bags) == 0:
			if self.bag_detect.has_stable == True:
				correct_bag_x, correct_bag_y = self.bag_detect.get_predict()
				cv2.rectangle(perspective_img, (correct_bag_x - 20, correct_bag_y - 20),
				              (correct_bag_x + 20, correct_bag_y + 20),
				              (30, 144, 255), 3)
				cv2.putText(perspective_img, "bag:({},{})".format(correct_bag_x, correct_bag_y),
				            (correct_bag_x - 20, correct_bag_y - 25),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 225), 2)
		else:
			for bag in bags:
				if abs(bag.cent_x - int(correct_bag_x)) < bag_real_width and abs(
						bag.cent_y - int(correct_bag_y)) < bag_real_width:
					correct_bag_x, correct_bag_y = bag.cent_x, bag.cent_y
					self.bag_detect.load_or_update_position((correct_bag_x, correct_bag_y))
					break
			else:
				if self.bag_detect.has_stable == True:
					correct_bag_x, correct_bag_y = self.bag_detect.get_predict()
					cv2.rectangle(perspective_img, (correct_bag_x - 20, correct_bag_y - 20),
					              (correct_bag_x + 20, correct_bag_y + 20),
					              (30, 144, 255), 3)
					cv2.putText(perspective_img, "bag:({},{})".format(correct_bag_x, correct_bag_y),
					            (correct_bag_x - 20, correct_bag_y - 25),
					            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 225), 2)

		return correct_bag_x, correct_bag_y

	def move_y_direct(self, abs_y_distance, correct_bag_x, correct_bag_y, down, hock_x, hock_y, north, perspective_img,
	                  south, up, y_distance):
		self.y_distance.pop('w' if y_distance > 0 else 'e')
		if abs_y_distance > permissible_distance:

			west = abs_y_distance if y_distance > 0 else 0
			east = abs_y_distance if y_distance < 0 else 0
			self.write_move_instruct(correct_bag_x, correct_bag_y, down, east, hock_x, hock_y, north,
			                         perspective_img, south, up, west)
			self.plchandle.move_status(0)
		else:
			self.keep_y_move = False
			self.plchandle.move_status(0)

	def write_move_instruct(self, correct_bag_x, correct_bag_y, down, east, hock_x, hock_y, north, perspective_img,
	                        south, up, west):
		cv2.putText(perspective_img, "hock:{},{}".format(hock_x, hock_y), (500, 400),
		            cv2.FONT_HERSHEY_SIMPLEX,
		            1.2,
		            (255, 255, 255), 2)

		cv2.putText(perspective_img, "bag:{},{}".format(correct_bag_x, correct_bag_y), (500, 500),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		            (255, 255, 255), 2)

		cv2.putText(perspective_img, "moving:{}".format(self.plchandle.read_status()), (500, 600),
		            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		            (255, 255, 255), 2)
		# print("move status is {}".format(self.plchandle.read_status()))
		if self.plchandle.read_status() == 0:
			self.plchandle.move(east=east, west=west, south=south, nourth=north, up=0, down=0)

	def move_close_bag_withoutlandmark(self, correct_bag_x=None, correct_bag_y=None, show=None, compensate=False):
		if compensate == False:
			self.loss_landmark_warn(correct_bag_x, correct_bag_y, show)

		return show

	def loss_landmark_warn(self, correct_bag_x, correct_bag_y, show):
		self.retry_find_landmark += 1
		self.move_when_misslandmark()
		if self.retry_find_landmark < move_tobag_retry_times:
			cv2.putText(show, "warn:landmark miss", (500, 400), cv2.FONT_HERSHEY_SIMPLEX,
			            1.2,
			            (255, 255, 255), 2)
			cv2.putText(show, "bag:{},{}".format(correct_bag_x, correct_bag_y), (500, 500),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			self.send_warn_info.emit("地标丢失，为了精确建议向前手动位移一段距离，找回地标")

		else:
			self.send_warn_info.emit("初步移动到指定位置,地标找回重试次数已到，精确定位结束")
			cv2.putText(show, "finish", (500, 400),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			cv2.putText(show, "bag:{},{}".format(correct_bag_x, correct_bag_y), (500, 500),
			            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			self.keep_x_move, self.keep_y_move = False, False
			self.retry_find_landmark = 0

	# ------------------------------------------------
	# 名称：compute_img
	# 功能：处理视频帧处理
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def compute_img(self, show, index):
		plc_connect = self.plc_connect()

		if plc_connect == False:
			logger("PLC连接失败", level='info')
			cv2.putText(show, "plc connect fail", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			return show

		perspective_img, find_landmark = self.landmark_detect.position_landmark(show)

		if find_landmark == False:

			bag_info = self.get_currentbag_position_withoutlandmark(show)  # 袋子位置信息
			hock_info = self.get_hock_position_withoutlandmark(show)  # 钩子位置信息
			if self.current_bag is not None and self.current_bag.status_map[
				'drop_hock'] == True and self.current_bag.step == 'drop_hock':
				self.suck_bag(perspective_img=None, original_img=show, frameindex=index)
			else:
				self.vision_no_landmark(msg="没有发现地标，向东移动1米")
			return show
		else:
			self.update_hockposition(perspective_img, original_img=None, find_landmark=find_landmark)

			self.choose_or_update_currentbag(perspective_img, original_img=None, find_landmark=find_landmark)

			if self.current_bag is not None and self.hock is not None:
				self.move_or_suck(perspective_img=perspective_img, original_img=show, frameindex=index)
			else:
				self.vision_no_bag(msg="没有发现袋子，向东移动1米")

			return perspective_img

	# ------------------------------------------------
	# 名称：move_or_suck
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def move_or_suck(self, perspective_img, original_img, frameindex):

		# print("hock position({},{}),bag position({},{})".format(self.hock.x, self.hock.y, self.current_bag.x,
		#                                                         self.current_bag.y))
		if abs(
				self.current_bag.x - self.hock.x) > 50 or abs(
			self.current_bag.y - self.hock.y - HOCK_DISTANCE) > 50:
			self.move_to_nearestbag(perspective_img)
		else:

			if self.current_bag.step == 'move_close':
				self.current_bag.step = 'drop_hock'
				self.current_bag.status_map['drop_hock'] = True
			if self.current_bag.step == 'drop_hock' and self.current_bag.status_map['drop_hock'] == True:
				self.suck_bag(perspective_img=None, original_img=original_img, frameindex=frameindex)

	# ------------------------------------------------
	# 名称：vision_no_landmark
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def vision_no_landmark(self, msg=None):
		if DEBUG:
			return
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()

			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')


		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	# ------------------------------------------------
	# 名称：vision_no_bag
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def vision_no_bag(self, msg=None):
		if DEBUG:
			return
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()

			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')

		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	# ------------------------------------------------
	# 名称：find_laster
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def find_laster(self, dest, find_landmark=False):

		dest_copy = dest.copy()
		laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		                                                               middle_end=450)
		if laster is not None and find_landmark == True:
			self.last_laster = laster
		self.laster = laster
		return dest

	# ------------------------------------------------
	# 名称：hock_moveto_center
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def hock_moveto_center(self):
		try:
			self.plchandle.move(nourth=2)
		except:
			logger("plc is not connect", level="error")

	# ------------------------------------------------
	# 名称：update_hockposition
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def update_hockposition(self, perspective_img=None, original_img=None, find_landmark=False):
		if find_landmark == False:
			return None
		perspective_img_copy = perspective_img.copy()

		hock, hock_foreground = self.hock_detect.location_hock_withlandmark(perspective_img, perspective_img_copy,
		                                                                    self.laster_status,
		                                                                    middle_start=middle_start_withlandmark,
		                                                                    middle_end=middle_end_withlandmark)

		if hock is None:
			return None

		self.last_hock = hock
		if hock is not None:
			if self.hock is None or not hasattr(self, 'hock'):
				self.hock = hock
			else:
				self.hock.center_x, self.hock.center_y = hock.center_x, hock.center_y

			self.hock.modify_box_content()
			cv2.rectangle(perspective_img, (self.hock.center_x - 20, self.hock.center_y - 20),
			              (self.hock.center_x + 20, self.hock.center_y + 20),
			              (255, 0, 0), 3)

			cv2.putText(perspective_img, self.hock.box_content,
			            (self.hock.center_x - 20, self.hock.center_y - 25),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		else:
			self.hock = None

	# ------------------------------------------------
	# 名称：get_currentbag_position_withoutlandmark
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def get_currentbag_position_withoutlandmark(self, original_img=None):

		if self.current_bag is None:
			return None

		bags, _foreground = self.bag_detect.location_bags_withoutlandmark(original_img)
		if bags is None or len(bags) == 0:
			return None
		# if self.current_bag is None:
		# self.current_bag = bags[0]
		# self.current_bag.step = 'drop_hock'
		# self.current_bag.status_map['drop_hock'] = True
		choose_bag = list(filter(lambda bag: bag.id == self.current_bag.id, bags))
		if choose_bag is None or len(choose_bag) == 0:
			return None
		else:
			return choose_bag[0].x, choose_bag[0].y

	# ------------------------------------------------
	# 名称：get_hock_position_withoutlandmark
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def get_hock_position_withoutlandmark(self, original_img=None):
		hock, _foreground = self.hock_detect.location_hock_withoutlandmark(original_img)
		if hock is None:
			return None
		else:
			x, y, w, h = hock.x, hock.y, hock.w, hock.h
			cv2.rectangle(original_img, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 255, 0), 1)
			return hock.x, hock.y

	# ------------------------------------------------
	# 名称：choose_or_update_currentbag
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def choose_or_update_currentbag(self, perspective_img=None, original_img=None, find_landmark=False):
		if find_landmark == False:
			return None
		if not DEBUG:
			ugent_stop_status = self.plchandle.is_ugent_stop()
		else:
			ugent_stop_status = 0

		if ugent_stop_status == 1:
			self.landmark_detect.draw_grid_lines(perspective_img if perspective_img is not None else original_img)
			return None

		perspective_copy_img = perspective_img.copy()

		bags, bag_forground = self.bag_detect.location_bags_withlandmark(perspective_img, perspective_copy_img,
		                                                                 find_landmark,
		                                                                 middle_start=120,
		                                                                 middle_end=500)
		self.bags = bags

		if self.current_bag is not None:
			for bag in self.bags:
				if bag.id == self.current_bag.id and self.current_bag.status_map['finish_move'] == False:
					self.current_bag = bag

		if self.current_bag and self.current_bag.status_map['finish_move'] == False:
			return None

		if bags is None or len(bags) == 0:
			if self.hock is not None:
				if self.hock.y < 100:
					self.finish_job = True
					self.plchandle.ugent_stop = True

					self.plchandle.power = False
			else:
				# 袋子检测失败
				self.vision_no_landmark(msg="没有发现袋子，向东移动一米")
				self.landmark_detect.draw_grid_lines(perspective_img)
			return None
		else:
			need_process_bags = [bag for bag in self.bags if bag.status_map['finish_move'] == False]

			if need_process_bags is None or len(need_process_bags) == 0:
				self.finish_job = True
				self.plchandle.power = False
				return None

		if self.hock is None:
			self.landmark_detect.draw_grid_lines(perspective_img if perspective_img is not None else original_img)
			return None

		if self.current_bag is None or self.current_bag.status_map['finish_move'] == True:
			choose_index = self.choose_nearest_bag(self.bags, self.hock)
			choosed_bag = bags[choose_index]
			self.current_bag = choosed_bag
			self.current_bag.status_map['choose'] = True
			self.current_bag.step = 'choose'
			logger("find nearest bag->({},{})".format(choosed_bag.x, choosed_bag.y), level='info')

	# ------------------------------------------------
	# 名称：down_hock
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def down_hock(self, much=50):
		self.plchandle.move(down=much)
		if self.current_bag is not None:
			self.current_bag.down_hock_much += much

	# ------------------------------------------------
	# 名称：check_suck
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def check_suck(self, original_img=None, frame_index=0):
		suck_success = False

		bag_info = self.get_currentbag_position_withoutlandmark(original_img)

		if bag_info is None:
			return False

		bag_col, bag_row = bag_info

		if self.current_bag.previous_position is None:
			self.current_bag.previous_position = [bag_col, bag_row]

		x_change = abs(bag_col - self.current_bag.previous_position[0])
		y_change = abs(bag_row - self.current_bag.previous_position[1])

		if 20 > x_change > 3 or 20 > y_change > 3:

			hock_info = self.get_hock_position_withoutlandmark(original_img)
			if hock_info is None: return False
			hock_col, hock_row = hock_info
			if abs(hock_col - bag_col) < 10 and abs(hock_row - bag_row) < 10:
				print("已经吸住袋子")
				return True
			else:
				return False

	# ------------------------------------------------
	# 名称：check_hold
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def check_hold(self, perspective_img, original_img, index=0):
		# cv2.putText(dest, "check_hold", (360, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		#             (255, 255, 255), 2)

		# self.suck_times = 0
		if self.status_show is not None:
			self.status_show.showMessage("检测是否钩住袋子")
		return True

	# ------------------------------------------------
	# 名称：pull_bag
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def pull_bag(self):
		'''
		拉起袋子，放到目的区域，目前没做
		:return:
		'''
		pass

	# ------------------------------------------------
	# 名称：suck_bag
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def suck_bag(self, perspective_img=None, original_img=None, frameindex=0):

		if self.current_bag is not None and self.current_bag.status_map['hock_suck'] == True:
			return True

		if perspective_img is not None:
			cv2.putText(perspective_img, "droping hock", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
		else:
			cv2.putText(original_img, "droping hock", (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

		if self.status_show is not None:
			self.status_show.showMessage("定位钩下降")

		if_suck, max_drop_z, has_droped_z = False, 400, 0  # 捡起
		while if_suck == False and has_droped_z < max_drop_z:
			self.down_hock(100)
			has_droped_z += 100
			if perspective_img is not None:
				self.move_to_nearestbag(perspective_img)
			else:
				if_suck = self.check_suck(original_img=original_img, frame_index=frameindex)
		if if_suck == True:
			# print(self.current_bag.suck_frame_status)
			self.current_bag.step = 'hock_suck'
			self.current_bag.status_map['hock_suck'] = True
			# print(self.current_bag.suckhock_positions)
			if perspective_img is not None:
				cv2.putText(perspective_img, "suck bag success", (500, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			if original_img is not None:
				cv2.putText(original_img, "suck bag success", (500, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)

			if self.status_show is not None:
				self.status_show.showMessage("定位钩已吸住" if if_suck else "未吸住")
			hold_bag = self.check_hold(perspective_img, original_img, frameindex)
			if hold_bag:
				self.pull_bag()

	# ------------------------------------------------
	# 名称：suck_bag
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def move_to_nearestbag(self, perspective_img):
		'''
		钩子向袋子靠近
		:param dest:
		:return:
		'''
		if perspective_img is None:
			return None
		try:
			if not DEBUG:
				move_status = self.plchandle.read_status()
				is_ugent_stop = self.plchandle.is_ugent_stop()
			else:
				move_status = 0
				is_ugent_stop = 0

			# move==1说明行车在移动中，0静止
			if move_status == 1 or is_ugent_stop == 1:
				if len(self.input_move_instructs) > 0:
					cv2.putText(perspective_img, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX,
					            1.2,
					            (255, 255, 255), 2)

				self.landmark_detect.draw_grid_lines(perspective_img)
				return perspective_img

			current_car_x, current_car_y, current_car_z = self.hock.x, self.hock.y + HOCK_DISTANCE, 0

			# 写入目标坐标
			target_x, target_y, target_z = self.current_bag.x, self.current_bag.y, 0
			target_info = "bag_X:{},bag_Y:{}".format(target_x, target_y)
			logger(target_info, level='info')
			cv2.putText(perspective_img, target_info, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			east, west, south, north, up, down = 0, 0, 0, 0, 0, 0

			move_info = "I:"
			if target_x - current_car_x > 0:
				south = abs(target_x - current_car_x)
				move_info += "to S {} cm".format(south)
			else:
				north = abs(target_x - current_car_x)
				move_info += "to N {} cm".format(north)

			if target_y - current_car_y > 0:
				west = abs(target_y - current_car_y)
				move_info += ", to W {} cm".format(west)
			else:
				east = abs(target_y - current_car_y)
				move_info += ",to E {} cm".format(east)

			if target_z - current_car_z > 0:
				up = abs(target_z - current_car_z)
				move_info += ",UP {} cm".format(up)
			else:
				down = abs(target_z - current_car_z)
				move_info += ",DOWN {} cm".format(down)

			if not DEBUG:
				self.plchandle.move(east=east, west=west, south=south, nourth=north, up=up, down=down)
			logger(move_info, level='info')
			self.input_move_instructs.append(move_info)

			self.current_bag.status_map['move_close'] = True
			self.current_bag.step = 'move_close'

			if len(self.input_move_instructs) > 0:
				cv2.putText(perspective_img, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			current_hock_info = "HOCK->X:{},Y:{}".format(current_car_x, current_car_y)
			logger(current_hock_info, 'info')
			cv2.putText(perspective_img, current_hock_info, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			error_info = "ERROR:{},{},{}".format(abs(target_x - current_car_x), abs(target_y - current_car_y - 30),
			                                     abs(target_z - current_car_z))
			logger(error_info, level='info')
			cv2.putText(perspective_img, error_info, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			if not DEBUG:
				# 超出边界就要紧急停止
				if current_car_x < 100 or current_car_x > 550 or current_car_y < 20:
					self.ugent_stop_car(current_car_x, current_car_y, current_car_z, perspective_img)
		except Exception as e:
			# print(e)
			self.landmark_detect.draw_grid_lines(perspective_img)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
			return perspective_img
		self.landmark_detect.draw_grid_lines(perspective_img)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
		return perspective_img

	# ------------------------------------------------
	# 名称：ugent_stop_car
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def ugent_stop_car(self, current_car_x, current_car_y, current_car_z, dest=None):
		# 智能识别紧急停止行车
		if current_car_y == 0 or current_car_y > 800 or current_car_x == 0 or current_car_x > 500 or current_car_x < 0 or current_car_y < 0:

			if dest is not None:
				cv2.putText(dest, " ugent_stop {},{},{}".format(current_car_x, current_car_y, current_car_z),
				            (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			self.work = False
			if self.save_video:
				self.update_savevideo.emit(self.save_video_name)
			if not DEBUG:
				self.plchandle.ugent_stop()

	# ------------------------------------------------
	# 名称：choose_nearest_bag
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-12
	# ------------------------------------------------
	def choose_nearest_bag(self, bags, hock):

		def __compute_distance(bag, hock):
			# start= time.perf_counter()
			X_2 = math.pow(bag.x - hock.x, 2)
			Y_2 = math.pow(bag.y - hock.y, 2)
			distance = math.sqrt(X_2 + Y_2)
			# end = time.perf_counter()
			# print("distance is {},compute cost:{}".format(distance,end-start))
			return distance

		distances = [__compute_distance(bag, hock) for bag in bags if bag.status_map['finish_move'] == False]

		min_distance, choose_index = 10000, 0
		for index, d in enumerate(distances):
			if d < min_distance:
				min_distance = d
				choose_index = index
		return choose_index



class ProcessThread(QThread):
	update_savevideo = pyqtSignal(str)
	move_to_bag_signal = pyqtSignal(tuple)
	ariver_advice = pyqtSignal(str, str)
	add_scan_bag_signal = pyqtSignal(list)
	error_show_signal = pyqtSignal(dict)

	def __init__(self, video_player, IMGHANDLE=None, PLCHANDLE=None, parent=None, dock_img_player=None):
		super().__init__(parent=parent)
		self._playing = True
		self.test_mode = False
		self._finish = False
		self._scan = False
		self._grabbag = False
		self._putdownbag = False
		self._movetobag_x = True
		self._movetobag_y = True
		self.move_close = False
		self.error_info = {}
		self.video_player = video_player
		self.dock_img_player = dock_img_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		# self.plchandle = PLCHANDLE
		self.save_video = False
		self.out = None
		self.detectorhandle = DetectorHandle(PLCHANDLE)
		self._target_bag_position = []
		self.last_target_bag_position = None
		self.last_arive_bag_position = None

	@property
	def target_bag_position(self):
		return self._target_bag_position

	@target_bag_position.setter
	def target_bag_position(self, value):
		self._target_bag_position = value

	@property
	def grab_bag(self):
		return self._grabbag

	@grab_bag.setter
	def grab_bag(self, value):
		self._grabbag = value

	@property
	def putdown_bag(self):
		return self._putdownbag

	@putdown_bag.setter
	def putdown_bag(self, value):
		self._putdownbag = value

	def __del__(self):
		self._scan = False
		if hasattr(self.IMAGE_HANDLE, 'release') and self.IMAGE_HANDLE:
			self.IMAGE_HANDLE.release()

	@property
	def play(self):
		return self._playing

	# 启动播放
	@play.setter
	def play(self, value=True):
		self._playing = value

	@property
	def scan_bag(self):
		return self._scan

	@scan_bag.setter
	def scan_bag(self, value):
		self._scan = value

	@property
	def move_to_bag_x(self):
		return self._movetobag_x

	@move_to_bag_x.setter
	def move_to_bag_x(self, value):
		self._movetobag_x = value

	@property
	def move_to_bag_y(self):
		return self._movetobag_y

	@move_to_bag_y.setter
	def move_to_bag_y(self, value):
		self._movetobag_y = value

	def run(self):
		# 启动行车，最好先将定位钩移到中间
		self.detectorhandle.hock_moveto_center()
		save_video_name = time.strftime("%Y%m%d%X", time.localtime()).replace(":", "")
		self.save_video_name = "saved_" + save_video_name + '.avi'

		gray = np.zeros((400, 300))

		# index = 0
		while self.play and self.IMAGE_HANDLE:
			# sleep(1 / 13)
			# index += 1
			show = self.IMAGE_HANDLE.read()
			if show is None or not np.any(show):
				continue

			# if self.save_video == True:
			# 	if self.out is None:
			# 		self.out = cv2.VideoWriter(os.path.join(SAVE_VIDEO_DIR, self.save_video_name), fourcc, 20.0,
			# 		                           (900, 700))
			# 	self.out.write(show)

			rows, cols, channels = show.shape
			if rows != IMG_HEIGHT or cols != IMG_WIDTH:
				show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
			else:
				show = show


			if self.scan_bag == True:
				show = self.scan_bags(show)


			if self.move_close == True and self.target_bag_position is not None and len(self.target_bag_position) > 0:
				movestatus = self.detectorhandle.plchandle.read_status()


				if self.last_target_bag_position is None or len(self.last_target_bag_position) == 0:
					self.last_target_bag_position = self.target_bag_position
					self.move_to_bag_signal.emit((self.target_bag_position[0], self.target_bag_position[1]))
				else:
					last_x, last_y, current_x, current_y = self.last_target_bag_position[0], \
					                                       self.last_target_bag_position[1], \
					                                       self.target_bag_position[0], \
					                                       self.target_bag_position[1]
					if abs(int(last_x) - int(current_x)) > 20 or abs(int(last_y) - int(current_y)) > 20:
						self.move_to_bag_signal.emit((self.target_bag_position[0], self.target_bag_position[1]))

				if self.move_to_bag_x == False and self.move_to_bag_y == False:
					cv2.putText(show, "arive", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					            (255, 255, 255), 2)
					cv2.putText(show, "hock:{},{}".format(self.detectorhandle.modify_hockposition[0],
					                                      int(self.detectorhandle.modify_hockposition[1])
					                                      ), (300, 400),
					            cv2.FONT_HERSHEY_SIMPLEX,
					            1.2,
					            (255, 255, 255), 2)

					bag_height = self.get_height(show)

					cv2.putText(show, "bag:{},{},{}".format(self.detectorhandle.modify_bagposition[0],
					                                        self.detectorhandle.modify_bagposition[1], bag_height),
					            (300, 500),
					            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					            (255, 255, 255), 2)

					cv2.putText(show, "moving:{}".format(self.detectorhandle.plchandle.read_status()), (300, 600),
					            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					            (255, 255, 255), 2)
					logger("arrive,bag:({},{},{}),hock:({},{})".format(self.detectorhandle.modify_bagposition[0],
					                                                   self.detectorhandle.modify_bagposition[1],
					                                                   bag_height,
					                                                   self.detectorhandle.modify_hockposition[0],
					                                                   int(self.detectorhandle.modify_hockposition[
						                                                       1])), level="info")

					if self.last_arive_bag_position is None:
						self.last_arive_bag_position = self.target_bag_position
						self.ariver_advice.emit(
							self.target_bag_position[0],
							self.target_bag_position[1])

				# self.lowing_and_grab(start=False, detect_handle=self.detectorhandle)
				elif movestatus == 0 or movestatus == "0":

					# if self.detectorhandle.slow == True: sleep(4)
					# cv2.putText(show, "wait for hock stop",
					#             (200, 600),
					#             cv2.FONT_HERSHEY_SIMPLEX, 1,
					#             (0, 255, 255), 1)
					sleep(WAIT_TIME)  # 等待射灯以及定位钩子晃动停止下来，目前来看不仅仅3分钟
					show, gray = self.move_instructs(show)

			self.show_height(show)


			plc_connect = self.detectorhandle.plc_connect()
			# if plc_connect == True or DEBUG == True:
			self.error_info = self.detectorhandle.plc_check_errors()
			if len(self.error_info.items()) > 0:
				self.error_show_signal.emit(self.error_info)

			now_rows, now_cols, channels = show.shape
			if now_rows != IMG_HEIGHT or now_cols != IMG_WIDTH:
				show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))

			dest = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

			finalimg = QImage(dest.data, dest.shape[1], dest.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(finalimg))
			self.video_player.setScaledContents(True)

			if self.dock_img_player is not None and gray is not None:
				finaldirectimg = QImage(gray.data, gray.shape[1], gray.shape[0], QImage.Format_RGB888)
				self.dock_img_player.setPixmap(QPixmap.fromImage(finaldirectimg))
				self.dock_img_player.setScaledContents(True)

		try:
			self.plchandle.reset()
		except:
			logger("PLC重置失败", 'error')

	def show_height(self, show):

		try:
			height = self.detectorhandle.plchandle.get_high()
		except Exception as e:
			height = "h:error"
			cv2.putText(show, height,
			            (500, 100),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		else:
			cv2.putText(show, "h:{}".format(height),
			            (510, 100),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

	def get_height(self, show):

		try:
			height = self.detectorhandle.plchandle.get_high()
		except Exception as e:
			height = None
		return height


	def scan_bags(self, show):
		show, temp_bag_positions, error_info = self.detectorhandle.scan_bags(show)
		if len(temp_bag_positions) > 0:
			self.add_scan_bag_signal.emit(temp_bag_positions)
		if len(error_info.items()) > 0:
			self.error_show_signal.emit(error_info)
		return show

	def move_instructs(self, show):
		show, hock_position, keep_x_move, keep_y_move, error_info = \
			self.detectorhandle.move_to_targetbag(show, self.target_bag_position, self.move_to_bag_x,
			                                      self.move_to_bag_y)
		self.move_to_bag_y = keep_y_move
		self.move_to_bag_x = keep_x_move
		self.error_info = {key: value for key, value in chain(self.error_info.items(), error_info.items())}
		if self.detectorhandle.hock is None or not hasattr(self.detectorhandle, 'hock'):
			print("hock detect fail")
			return show, None
		gray = np.zeros_like(show)
		# 上下糊弄事 老实人 吃亏啊
		# hock_x, hock_y = self.detectorhandle.hock.center_x, self.detectorhandle.hock.center_y
		# cv2.arrowedLine(show, (hock_x, hock_y), (int(self.target_bag_position[0]), int(self.target_bag_position[1])),
		#                 (255, 0, 255), thickness=3)
		# gray.resize(400,400)

		return show, gray

	def contact_pressure(self):
		pass

	def contactclose_imagedetect(self):
		return True

	def grab_banding(self):
		pass

	def up_much(self):
		bags = sorted(self.detectorhandle.bags, key=lambda item: item.height, reverse=False)
		min, height = bags[0].height, bags[len(bags) - 1].height
		return height - min + 1

	def putdown_bag(self):
		pass

	def reach_placementarea(self, img, current_position):

		find_placementarea, x, y, w, h = self.find_placementarea(img)
		hock_x, hock_y = current_position

		detecthandle = LandMarkDetecotr()
		hock_in_placementarea = detecthandle.point_in_rect(hock_x, hock_y, x, y, w, h)


		while find_placementarea or hock_in_placementarea == False:
			find_placementarea, x, y, w, h = self.find_placementarea(img)
			hock_in_placementarea = detecthandle.point_in_rect(hock_x, hock_y, x, y, w, h)
			if find_placementarea == True and hock_in_placementarea == True: break

		if find_placementarea and hock_in_placementarea:
			self.putdown_bag()

	def find_placementarea(self, img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		color_low, color_high = [11, 43, 46], [34, 255, 255]
		color1_min, color1_max = np.array(color_low), np.array(color_high)
		color1_mask = cv2.inRange(hsv, color1_min, color1_max)
		ret, foreground = cv2.threshold(color1_mask, 0, 255, cv2.THRESH_BINARY)
		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		find = contours is not None and len(contours) > 0 and ShapeDetector().detect(contours[0], Shape.RECTANGLE)

		# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
		# detect_result = ShapeDetector().detect(contours[0], Shape.RECTANGLE)
		#
		# cv2.putText(img, "{}".format(detect_result), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		#             (255, 100, 255), 2)

		if find == True:
			x, y, w, h = cv2.boundingRect(contours[0])
		else:
			x, y, w, h = 0, 0, 0, 0
		return find, x, y, w, h

	def lowing_and_grab(self, start=False, detect_handle=None):

		if start == False: return
		pressure_v = self.contact_pressure()
		pressure_threhold = 10

		is_contractclose = False
		while pressure_v < pressure_threhold or self.contactclose_imagedetect() == False:
			pressure_v = self.contact_pressure()
			if pressure_v >= pressure_threhold:
				is_contractclose = True
				break

		is_grab_banding = False
		if is_contractclose and is_grab_banding == False:
			while is_grab_banding == False:
				is_grab_banding = self.grab_banding()

			if is_grab_banding == True:
				up_height = self.up_much()

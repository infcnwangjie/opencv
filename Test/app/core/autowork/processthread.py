# -*- coding: utf-8 -*-
import math
import time
from time import sleep

import cv2
import numpy
from PyQt5.QtCore import QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

from app.config import IMG_WIDTH, IMG_HEIGHT
from app.core.exceptions.allexception import SdkException, NotFoundBagException, NotFoundHockException
from app.core.plc.plchandle import PlcHandle
from app.core.beans.locationservice import PointLocationService, BAG_AND_LANDMARK
from app.core.processers.bag_detector import BagDetector
from app.core.processers.landmark_detector import LandMarkDetecotr
from app.core.processers.laster_detector import LasterDetector
from app.log.logtool import mylog_error, mylog_debug
from app.status import HockStatus
import gevent


class ProcessThread(QThread):
	positionSignal = pyqtSignal(tuple)
	dropHockSignal = pyqtSignal(tuple)
	pullHockSignal = pyqtSignal(tuple)
	findConveyerBeltSignal = pyqtSignal(tuple)
	dropBagSignal = pyqtSignal(tuple)
	rebackSignal = pyqtSignal(tuple)
	finishSignal = pyqtSignal(str)
	foundbagSignal = pyqtSignal(int)

	def __init__(self, video_player, IMGHANDLE=None, PLCHANDLE=None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = True
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		self.plchandle = PLCHANDLE
		self.bags = []
		self.send_positions = []
		self.landmark_detect = LandMarkDetecotr()
		self.bag_detect = BagDetector()
		self.laster_detect = LasterDetector()
		self.history_bags = []
		self.history_laster_travel = []

	def __del__(self):
		self._working = False
		if hasattr(self.IMAGE_HANDLE, 'release') and self.IMAGE_HANDLE:
			self.IMAGE_HANDLE.release()



	@property
	def play(self):
		return self._playing

	@play.setter
	def play(self, value=True):
		self._playing = value

	@property
	def work(self):
		return self._working

	@work.setter
	def work(self, value=True):
		self._working = value

	def run(self):

		while self.play and self.IMAGE_HANDLE:
			# sleep(1 / 8)
			show = self.IMAGE_HANDLE.read()
			if show is None:
				break

			dest = self.compute_img(show) if self.work else show

			dest = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)
			finalimg = QImage(dest.data, dest.shape[1], dest.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(finalimg))
			self.video_player.setScaledContents(True)

	def compute_img(self, show):
		'''
		目标识别：控制逻辑部分
		:param show:
		:return:
		'''
		dest, find_landmark = self.landmark_detect.position_landmark(show)

		if not find_landmark:
			# 当前帧，地标定位失败
			return dest

		dest_copy = dest.copy()

		laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=250,
		                                                               middle_end=500)
		if laster is None:
			# 当前帧，钩子定位失败
			self.landmark_detect.draw_grid_lines(dest)
			return dest

		print("激光斑点坐标 is ({x},{y})".format(x=laster.x, y=laster.y))

		self.history_laster_travel.append((laster.x, laster.y))  # 记录激光灯移动轨迹，用来纠偏

		bags, bag_forground = self.bag_detect.location_bags(dest, dest_copy, find_landmark, middle_start=100,
		                                                    middle_end=400)
		if bags is None or len(bags) == 0:
			# 袋子检测失败
			self.landmark_detect.draw_grid_lines(dest)
			return dest

		if self.history_bags is not None and len(self.history_bags) > 0:
			last_time_bags = self.history_bags[-1]
			# 检测到的袋子忽多忽少的情况，一定是不稳定的；
			# 退出前记得绘制网格线
			if len(last_time_bags) != len(bags):
				self.landmark_detect.draw_grid_lines(dest)
				return dest

		choose_index = self.choose_nearest_bag(bags, laster)

		choosed_bag = bags[choose_index]
		print("will get to {},{}".format(choosed_bag.x, choosed_bag.y))

		try:
			move_status = self.plchandle.read_status()
			# move==1说明行车在移动中，0静止
			if move_status == 1:
				pass

			print("移动状态为：{}".format(move_status))
			# 视频中行车激光位置，钩子的位置需要定位

			# 写入目标坐标
			bag_z = 0
			plc_target_x, plc_target_y, plc_target_z = self.plchandle.target_position()
			if abs(plc_target_x - choosed_bag.x) > 20 or abs(plc_target_y - choosed_bag.y) < 20 or abs(
					plc_target_z - bag_z) < 20:
				self.plchandle.write_target_position([choosed_bag.x, choosed_bag.y, bag_z])

			# 写入钩子当前坐标
			# TODO current_car_z
			plc_x, plc_y, plc_z = self.plchandle.current_hock_position()
			current_car_x, current_car_y, current_car_z = laster.x, laster.y + 100, 0
			if abs(plc_x - current_car_x) > 20 or abs(plc_y - current_car_y) > 20 or abs(plc_z - current_car_z) > 20:
				self.plchandle.write_hock_position([current_car_x, current_car_y, current_car_z])

			# 不用写入纠偏量当前坐标
			# plc_target_x, plc_target_y, plc_target_z = self.plchandle.target_position()
			# if abs(plc_target_y - current_car_y) > 20 or abs(plc_target_x - current_car_x) > 20:
			# 	error_x = plc_target_x - current_car_x
			# 	error_y = plc_target_y - current_car_y
			# 	self.plchandle.write_error([error_x, error_y, 0])


		except:
			self.landmark_detect.draw_grid_lines(dest)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
			return dest

		self.landmark_detect.draw_grid_lines(dest)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
		return dest

	def choose_nearest_bag(self, bags, laster):
		'''
		选择距离钩子最近的袋子
		:param bags:
		:param laster:
		:return:
		'''

		def __compute_distance(bag, laster):
			'''
			choose_nearest_bag内部的计算袋子与钩子距离的方法
			:param bag:
			:param laster:
			:return:
			'''
			# start= time.perf_counter()
			X_2 = math.pow(bag.x - laster.x, 2)
			Y_2 = math.pow(bag.y - laster.y, 2)
			distance = math.sqrt(X_2 + Y_2)
			end = time.perf_counter()
			# print("distance is {},compute cost:{}".format(distance,end-start))

			return distance

		gent_list = []
		for index, bag in enumerate(bags):
			task = gevent.spawn(__compute_distance, bag, laster)
			gent_list.append(task)
		gevent.joinall(gent_list)

		min_distance, choose_index = 10000, 0
		for index, g in enumerate(gent_list):
			if g.value < min_distance:
				min_distance = g.value
				choose_index = index

		return choose_index

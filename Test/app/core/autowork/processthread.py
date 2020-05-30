# -*- coding: utf-8 -*-
import math
import os
import time
from time import sleep

import cv2
import numpy
from PyQt5.QtCore import QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

from app.config import IMG_WIDTH, IMG_HEIGHT, SAVE_VIDEO_DIR
from app.core.exceptions.allexception import SdkException, NotFoundBagException, NotFoundHockException
from app.core.plc.plchandle import PlcHandle
from app.core.processers.preprocess import LandMarkDetecotr, BagDetector, LasterDetector

from app.log.logtool import mylog_error, mylog_debug, logger
from app.status import HockStatus


# import gevent


class ProcessThread(QThread):
	'''
	目标识别程序主程
	'''

	current_target_x, current_target_y = 0, 0  # 当前目标袋子坐标 x,y
	current_hock_x, current_hock_y = 0, 0  # 当前钩子坐标 x,y
	error_x, error_y = 0, 0  # 当前x坐标纠偏，当前y坐标纠偏
	input_move_instructs = []  # 所有的移动指令
	hock_points = []
	update_savevideo = pyqtSignal(str)

	def __init__(self, video_player, IMGHANDLE=None, PLCHANDLE=None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = False
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
		self.save_video = False

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
		save_video_name = time.strftime("%Y%m%d%X", time.localtime()).replace(":", "")
		self.save_video_name = "saved_" + save_video_name + '.avi'
		out=None
		if self.save_video:
			fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 保存视频的编码
			out = cv2.VideoWriter(os.path.join(SAVE_VIDEO_DIR, self.save_video_name), fourcc, 20.0, (900, 700))

		# index = 0
		while self.play and self.IMAGE_HANDLE:
			sleep(1 / 13)
			# index += 1
			show = self.IMAGE_HANDLE.read()
			if show is None:
				# 程序执行结束要重置PLC
				try:
					self.plchandle.reset()
				except:
					# print("plc is not in use")
					logger("PLC连接失败", 'error')
					break
				if self.save_video:
					self.update_savevideo.emit(save_video_name)
				break

			rows, cols, channels = show.shape
			if rows != IMG_HEIGHT or cols != IMG_WIDTH:
				show = cv2.resize(show, (IMG_WIDTH, IMG_HEIGHT))
			else:
				show = show

			dest = self.compute_img(show) if self.work else show

			if self.save_video:
				if out is None:
					fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 保存视频的编码
					out = cv2.VideoWriter(os.path.join(SAVE_VIDEO_DIR, self.save_video_name), fourcc, 20.0, (900, 700))
				out.write(dest)

			dest = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)

			finalimg = QImage(dest.data, dest.shape[1], dest.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(finalimg))
			self.video_player.setScaledContents(True)

		if self.save_video:
			self.update_savevideo.emit(self.save_video_name)
		# 程序执行结束要重置PLC
		try:
			self.plchandle.reset()
		except:
			logger("PLC重置失败", 'error')

	def compute_img(self, show):
		'''
		目标识别：控制逻辑部分
		:param show:
		:return:
		'''
		dest, find_landmark = self.landmark_detect.position_landmark(show)

		if not find_landmark:
			self.find_objects_first(msg="没有发现地标，向东移动1米")
			return dest

		if self.plchandle.is_open():
			return self.processimg_plcopen(dest, find_landmark)
		else:
			return self.processimg_plcclose(dest, find_landmark)

	def find_objects_first(self, msg=None):
		'''
		看不到地标或者看不到目标，是因为还没有进入视野
		:return:
		'''
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()
			# 确保行车没有被紧急停止或者没有在运动状态中
			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')
		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	def processimg_plcclose(self, dest, find_landmark):
		'''
		当PLC关闭的时候处理图像
		:param dest:
		:param find_landmark:
		:return:
		'''
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

		# 视频中行车激光位置，钩子的位置需要定位
		current_car_x, current_car_y, current_car_z = laster.x, laster.y + 100, 0

		# 写入目标坐标
		target_x, target_y, target_z = choosed_bag.x, choosed_bag.y, 0
		target_info = "bag_X:{},bag_Y:{}".format(target_x, target_y)
		cv2.putText(dest, target_info, (460, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
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
			move_info += ",to W {} cm".format(west)
		else:
			east = abs(target_y - current_car_y)
			move_info += ",to E {} cm".format(east)

		if target_z - current_car_z > 0:
			up = abs(target_y - current_car_y)
			move_info += ",UP {} cm".format(up)
		else:
			down = abs(target_y - current_car_y)
			move_info += ",DOWN {} cm".format(down)

		self.input_move_instructs.append(move_info)

		if len(self.input_move_instructs) > 0:
			cv2.putText(dest, self.input_move_instructs[-1], (460, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
		current_hock_info = "HOCK->X:{},Y:{}".format(current_car_x, current_car_y)
		cv2.putText(dest, current_hock_info, (460, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		            (255, 255, 255), 2)

		cv2.putText(dest, move_info, (460, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
		print("will get to {},{}".format(choosed_bag.x, choosed_bag.y))
		return dest

	def processimg_plcopen(self, dest, find_landmark):
		'''
		当PLC开启的时候处理图像
		:param dest:
		:param find_landmark:
		:return:
		'''
		ugent_stop_status = self.plchandle.is_ugent_stop()
		if ugent_stop_status == 1:
			self.landmark_detect.draw_grid_lines(dest)
			return dest
		dest_copy = dest.copy()
		laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=250,
		                                                               middle_end=500)
		if laster is None:
			# 当前帧，钩子定位失败
			self.landmark_detect.draw_grid_lines(dest)
			return dest

		logger("激光斑点坐标 is ({x},{y})".format(x=laster.x, y=laster.y), 'info')
		self.history_laster_travel.append((laster.x, laster.y))  # 记录激光灯移动轨迹，用来纠偏
		bags, bag_forground = self.bag_detect.location_bags(dest, dest_copy, find_landmark, middle_start=100,
		                                                    middle_end=400)
		if bags is None or len(bags) == 0:
			# 袋子检测失败
			self.find_objects_first(msg="没有发现袋子，向东移动一米")
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
		logger("find nearest bag->({},{})".format(choosed_bag.x, choosed_bag.y), level='info')
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()

			# move==1说明行车在移动中，0静止
			if move_status == 1 or is_ugent_stop == 1:
				if len(self.input_move_instructs) > 0:
					cv2.putText(dest, self.input_move_instructs[-1], (460, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					            (255, 255, 255), 2)
				# 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
				self.landmark_detect.draw_grid_lines(dest)
				return dest

			# 视频中行车激光位置，钩子的位置需要定位
			current_car_x, current_car_y, current_car_z = laster.x, laster.y + 100, 0
			# self.hock_points.append(current_car_x, current_car_y)

			# 写入目标坐标
			target_x, target_y, target_z = choosed_bag.x, choosed_bag.y, 0
			target_info = "bag_X:{},bag_Y:{}".format(target_x, target_y)
			logger(target_info, level='info')
			cv2.putText(dest, target_info, (460, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
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

			self.plchandle.move(east=east, west=west, south=south, nourth=north, up=up, down=down)
			logger(move_info, level='info')
			self.input_move_instructs.append(move_info)

			if len(self.input_move_instructs) > 0:
				cv2.putText(dest, self.input_move_instructs[-1], (460, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			current_hock_info = "HOCK->X:{},Y:{}".format(current_car_x, current_car_y)
			logger(current_hock_info, 'info')
			cv2.putText(dest, current_hock_info, (460, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			error_info = "ERROR:{},{},{}".format(abs(target_x - current_car_x), abs(target_y - current_car_y),
			                                     abs(target_z - current_car_z))
			logger(error_info, level='info')
			cv2.putText(dest, error_info, (460, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			# 智能识别紧急停止行车
			if current_car_y == 0 or current_car_y > 800 or current_car_x == 0 or current_car_x > 500 or current_car_x < 0 or current_car_y < 0:

				cv2.putText(dest, " ugent_stop {},{},{}".format(current_car_x, current_car_y, current_car_z),
				            (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
				self.work = False
				if self.save_video:
					self.update_savevideo.emit(self.save_video_name)
				self.plchandle.ugent_stop()


		except Exception as e:
			print(e)
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

		distances = [__compute_distance(bag, laster) for bag in bags]

		min_distance, choose_index = 10000, 0
		for index, d in enumerate(distances):
			if d < min_distance:
				min_distance = d
				choose_index = index
		return choose_index

# -*- coding: utf-8 -*-
import math
import os
import time
from time import sleep

import cv2
import numpy
from PyQt5.QtCore import QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

from app.config import IMG_WIDTH, IMG_HEIGHT, SAVE_VIDEO_DIR, LASTER_HOCK_DISTANCE, HOCK_DISTANCE, DEBUG
from app.core.beans.models import Bag, Laster, Hock
from app.core.exceptions.allexception import SdkException, NotFoundBagException, NotFoundHockException
from app.core.plc.plchandle import PlcHandle
from app.core.processers.preprocess import LandMarkDetecotr, BagDetector, LasterDetector, HockDetector

from app.log.logtool import mylog_error, mylog_debug, logger
from app.status import HockStatus


class DetectorHandle(object):
	'''
	袋子处理表
	'''
	instance = None
	current_target_x, current_target_y = 0, 0  # 当前目标袋子坐标 x,y
	current_hock_x, current_hock_y = 0, 0  # 当前钩子坐标 x,y
	error_x, error_y = 0, 0  # 当前x坐标纠偏，当前y坐标纠偏
	input_move_instructs = []  # 所有的移动指令
	hock_points = []

	def __new__(cls, *args, **kwargs):
		if cls.instance is None:
			cls.instance = super().__new__(cls)
		return cls.instance

	def __init__(self, plchandle):
		self.bags = []  # 袋子列表
		self.current_bag: Bag = None  # 当前处理的袋子
		self.hock: Hock = None  # 钩子
		self.last_hock: Hock = None  # 最近一次检测到的hock
		self.laster: Laster = None  # 激光灯
		self.last_laster: Laster = None  # 最近一次检测到的激光灯
		self.finish_job = False  # 完成了所有搬运工作
		self.landmark_detect = LandMarkDetecotr()
		self.bag_detect = BagDetector()
		self.laster_detect = LasterDetector()
		self.hock_detect = HockDetector()
		self.history_bags = []
		self.history_laster_travel = []
		self.plchandle = plchandle

	# self.init_background()

	def plc_connect(self):
		plcconnect = self.plchandle.is_open()
		return plcconnect if not DEBUG else True

	# def init_background(self):
	# 	self.foreground = cv2.createBackgroundSubtractorMOG2()

	def compute_img(self, show):
		'''
		# 落钩时机判断需要执行
		:param show:
		:return:
		'''
		plc_connect = self.plc_connect()
		dest, success = self.landmark_detect.position_landmark(show)

		if plc_connect == False:
			logger("PLC连接失败", level='info')
			cv2.putText(dest, "plc connect fail", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			return dest

		# 地标定位失败，就下一帧
		if success == False:
			self.vision_no_landmark(msg="没有发现地标，向东移动1米")
			return dest

		# 计算最近的袋子,就能与钩子最近的袋子找到
		find_hock = self.find_hock(dest, success)
		# TODO while not find_hock:drop_down_hock()
		# 如果当前袋子为空,或者当前袋子执行完毕,需要切换目标袋子
		self.choose_or_update_currentbag(dest, success)

		if self.current_bag is not None and self.hock is not None:
			# 当选定袋子并且激光灯也定位到,并且钩子并没有抵达
			self.check_arive_withhock(dest)
		#
		# elif self.current_bag is not None and self.hock is None:
		# 	# 当前袋子不为空，激光斑点定位失败
		# 	self.check_arive_withouthock(dest)
		else:
			# 行车与目标距离较远，没有识别到
			self.vision_no_landmark(msg="没有发现袋子，向东移动1米")

		return dest

	def check_arive_withhock(self, dest):

		print("hock position({},{}),bag position({},{})".format(self.hock.x,self.hock.y,self.current_bag.x,self.current_bag.y))

		if abs(
				self.current_bag.x - self.hock.x) > 50 or abs(
			self.current_bag.y - self.hock.y - HOCK_DISTANCE) > 50:
			self.move_to_nearestbag(dest)
		else:
			self.arrive_bag(dest, has_close=True)

	def vision_no_landmark(self, msg=None):
		'''
		看不到地标或者看不到目标，是因为还没有进入视野
		:return:
		'''
		if DEBUG:
			return
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()
			# 确保行车没有被紧急停止或者没有在运动状态中
			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')
		# laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		#                                                                middle_end=450)
		# self.ugent_stop_car(laster.x, laster.y + 50, 0 )
		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	def find_laster(self, dest, find_landmark=False):
		'''
		找到激光灯
		:param dest:
		:return:
		'''
		dest_copy = dest.copy()
		laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		                                                               middle_end=450)
		if laster is not None and find_landmark == True:
			self.last_laster = laster
		self.laster = laster
		return dest

	# 找到定位钩
	def find_hock(self, dest, find_landmark=False):
		dest_copy = dest.copy()
		hock, hock_foreground = self.hock_detect.location_hock(dest, dest_copy, middle_start=120,
		                                                       middle_end=470)
		if hock is not None and find_landmark == True:
			self.last_hock = hock
		self.hock = hock
		return self.hock is None

	# 决定抓取哪个袋子
	def choose_or_update_currentbag(self, dest, find_landmark):
		if not DEBUG:
			ugent_stop_status = self.plchandle.is_ugent_stop()
		else:
			ugent_stop_status=0

		if ugent_stop_status == 1 or find_landmark == False:
			self.landmark_detect.draw_grid_lines(dest)
			return dest

		dest_copy = dest.copy()

		# 如果当前袋子不为空,并且当前袋子并没有完成任务,就直接返回该袋子好了,不过
		# 也有不妥的地方，假如前面的几帧因为角度问题，误差很大，还得需要计算袋子的定位
		# 因此这里牺牲了效率，追求准确
		bags, bag_forground = self.bag_detect.location_bags(dest, dest_copy, find_landmark, middle_start=120,
		                                                    middle_end=500)
		self.bags = bags

		if self.current_bag is not None:
			for bag in self.bags:
				if bag.id == self.current_bag.id and self.current_bag.status_map['finish_move'] == False:
					self.current_bag=bag
					# if abs(self.current_bag.y - bag.y) > 100:
						# 误差过大，八成是地标识别错误
						# break
					# else:
					# 	self.current_bag = bag
					# 	break

		if self.current_bag and self.current_bag.status_map['finish_move'] == False:
			return dest

		if bags is None or len(bags) == 0:
			if self.hock is not None:
				if self.hock.y < 100:
					self.finish_job = True  # 结束自动运输任务
					self.plchandle.ugent_stop = True  # 越界紧急停止运动
					self.plchandle.power = False  # 行车梯形图断电
			else:
				# 袋子检测失败
				self.vision_no_landmark(msg="没有发现袋子，向东移动一米")
				self.landmark_detect.draw_grid_lines(dest)
			return dest
		else:
			need_process_bags = [bag for bag in self.bags if bag.status_map['finish_move'] == False]
			if need_process_bags is None or len(need_process_bags) == 0:
				# 没有要移动的袋子,让plc关闭梯形图程序
				self.finish_job = True
				self.plchandle.power = False
				return dest

		if self.hock is None:
			# 当前帧，钩子定位失败
			self.landmark_detect.draw_grid_lines(dest)
			return dest

		# 如果当前没有需要运行的袋子，或者当前目标袋子已经处理完成，则换一个袋子
		if self.current_bag is None or self.current_bag.status_map['finish_move'] == True:
			choose_index = self.choose_nearest_bag(self.bags, self.hock)
			choosed_bag = bags[choose_index]
			self.current_bag = choosed_bag
			self.current_bag.status_map['choose'] = True
			self.current_bag.step = 'choose'
			logger("find nearest bag->({},{})".format(choosed_bag.x, choosed_bag.y), level='info')

		return dest

	# 放下钩子
	def down_hock(self, much=50):
		'''
		放下钩子
		:param much:
		:return:
		'''
		self.plchandle.move(down=much)
		if self.current_bag is not None:
			self.current_bag.down_hock_much += much

	# 	检查定位钩吸起袋子
	def check_suck(self,dest):
		'''
		袋子如果被定位钩抓起，尝试拉起袋子，会让袋子左右摇晃
		:return:
		'''
		# TODO
		cv2.putText(dest, "check_suck", (310, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					(255, 255, 255), 2)
		return True

	# 检查机械手抓住袋子
	def  check_hold(self,dest):
		'''
		检测机械手抓住袋子：袋子的面积会因为被机械手遮挡而变小
		:return:
		'''
		cv2.putText(dest, "check_hold", (360, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					(255, 255, 255), 2)
		return True

	# 拉起袋子
	def pull_bag(self):
		'''
		拉起袋子，放到目的区域，目前没做
		:return:
		'''
		pass

	def arrive_bag(self, dest, has_close=False, z=50):
		cv2.putText(dest, "droping hock", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		            (255, 255, 255), 2)
		self.current_bag.status_map['drop_hock'] = True
		self.current_bag.step = 'drop_hock'

		if_suck,max_drop_z,has_droped_z = False,600,0  # 捡起
		while not if_suck and has_droped_z<max_drop_z:
			self.down_hock(50)
			has_droped_z += 50
			self.move_to_nearestbag(dest)
			if_suck= self.check_suck(dest)
		# endwhile
		if if_suck == True:
			hold_bag=self.check_hold(dest)
			if hold_bag:
				self.pull_bag()




	# TODO 视频识别是否需要停止落钩
	def check_arive_withouthock(self, dest):
		'''
					if  检测到钩子:
						distance=computedistance(钩子，袋子)
						if distance < c1:
							while  abs(钩子.z,袋子.z)>z1:
								drop_hock(z=50cm)
								update(钩子.z,袋子.z)->
					else:
						while not find_hock:
							drop_hock(z=50cm)
							find_hock=detect_hock()
							if find_hock:
								break

						distance=computedistance(钩子，袋子)
						if distance< c1:
							while  abs(钩子.z,袋子.z)>z1:
								drop_hock(z=50cm)
								update(钩子.z,袋子.z)


					#如果检测不到钩子，判断最后一次出现，钩子的位置，并计算钩子的位置与目标袋子距离多少
					#如果距离够近，尝试落钩，直到摄像头能检测到
					'''
		if abs(self.last_hock.x - self.current_bag.x) < 100 and abs(
				self.last_hock.y - self.current_bag.y - HOCK_DISTANCE) < 100:
			# 最近一次检测到的激光灯已经接近目标袋子了,
			hock_has_drop = 0
			MAX_DROP_HOCK = 700
			find = False
			while hock_has_drop < MAX_DROP_HOCK and not find:
				self.down_hock(much=50)
				hock_has_drop += 50
				self.find_hock(dest=dest, find_landmark=True)
				if self.hock is not None:
					find = True
				if find == True or hock_has_drop > MAX_DROP_HOCK: break
			# end while
			if find == True:
				# TODO 落钩细节
				# self.arrive_bag(has_close=True)
				pass

		else:
			# 先放下钩子，检测到钩子，再判断距离袋子的距离做打算
			pass

	def move_to_nearestbag(self, dest):
		'''
		钩子向袋子靠近
		:param dest:
		:return:
		'''
		try:
			if not DEBUG:
				move_status = self.plchandle.read_status()
				is_ugent_stop = self.plchandle.is_ugent_stop()
			else:
				move_status=0
				is_ugent_stop=0

			# move==1说明行车在移动中，0静止
			if move_status == 1 or is_ugent_stop == 1:
				if len(self.input_move_instructs) > 0:
					cv2.putText(dest, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
					            (255, 255, 255), 2)
				# 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
				self.landmark_detect.draw_grid_lines(dest)
				return dest

			# 视频中行车激光位置，钩子的位置需要定位
			current_car_x, current_car_y, current_car_z = self.hock.x, self.hock.y + HOCK_DISTANCE, 0
			# self.hock_points.append(current_car_x, current_car_y)

			# 写入目标坐标
			target_x, target_y, target_z = self.current_bag.x, self.current_bag.y, 0
			target_info = "bag_X:{},bag_Y:{}".format(target_x, target_y)
			logger(target_info, level='info')
			cv2.putText(dest, target_info, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
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
				cv2.putText(dest, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			current_hock_info = "HOCK->X:{},Y:{}".format(current_car_x, current_car_y)
			logger(current_hock_info, 'info')
			cv2.putText(dest, current_hock_info, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			error_info = "ERROR:{},{},{}".format(abs(target_x - current_car_x), abs(target_y - current_car_y - 30),
			                                     abs(target_z - current_car_z))
			logger(error_info, level='info')
			cv2.putText(dest, error_info, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			if not DEBUG:
				self.ugent_stop_car(current_car_x, current_car_y, current_car_z, dest)
		except Exception as e:
			print(e)
			self.landmark_detect.draw_grid_lines(dest)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
			return dest
		self.landmark_detect.draw_grid_lines(dest)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
		return dest

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

	def choose_nearest_bag(self, bags, hock):
		'''
		选择距离钩子最近的袋子
		:param bags:
		:param laster:
		:return:
		'''

		def __compute_distance(bag, hock):
			'''
			choose_nearest_bag内部的计算袋子与钩子距离的方法
			:param bag:
			:param laster:
			:return:
			'''
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

	def __init__(self, video_player, IMGHANDLE=None, PLCHANDLE=None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = False
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		# self.plchandle = PLCHANDLE
		self.save_video = False
		self.detectorhandle = DetectorHandle(PLCHANDLE)

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
		out = None
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
				show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
			else:
				show = show

			dest = self.detectorhandle.compute_img(show) if self.work and not self.detectorhandle.finish_job else show

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

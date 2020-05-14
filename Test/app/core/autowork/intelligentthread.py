# -*- coding: utf-8 -*-
from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
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


class IntelligentThread(QThread):
	'''
	智能识别流程如下：
	定位地标，根据地标对摄像做视图转换；
	定位钩子，找出钩子的位置；
	移动工控机，定位所有袋子（这步不是必须）；找出所有袋子的实际位置
	识别袋子，总共有多少袋子，循环袋子列表，轮询移动袋子到传送带；一个过程走完，要复位；
		在轮询袋子过程中，要时刻检测钩子的位置，并时刻纠偏；纠偏结果，要写入到plc
	如果循环完所有的袋子，应该给出已完工信号；
	'''
	positionSignal = pyqtSignal(tuple)
	dropHockSignal = pyqtSignal(tuple)
	pullHockSignal = pyqtSignal(tuple)
	findConveyerBeltSignal = pyqtSignal(tuple)
	dropBagSignal = pyqtSignal(tuple)
	rebackSignal = pyqtSignal(tuple)
	finishSignal = pyqtSignal(str)
	foundbagSignal = pyqtSignal(int)

	def __init__(self, video_player, IMGHANDLE=None,
	             positionservice: PointLocationService = None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = False
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		self.positionservice = positionservice  # 指令处理器
		self._hockstatus = HockStatus.POSITION_NEARESTBAG  # 钩子状态会影响定位程序
		self.bags = []
		self.send_positions = []

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

	@property
	def hockstatus(self):
		return self._hockstatus

	@hockstatus.setter
	def hockstatus(self, value=HockStatus.POSITION_NEARESTBAG):
		# value must HockStatus enum
		self._hockstatus = value

	def run(self):

		while self.play and self.IMAGE_HANDLE:
			sleep(1 / 13)
			frame = self.IMAGE_HANDLE.read()
			if frame is None:
				break
			if frame.ndim == 3:
				show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.ndim == 2:
				show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

			if self.work:
				self.positionservice.img = show
				self.core_process(show)
				show = self.positionservice.img
			self.video_player.set_img(show)

	def core_process(self, img):
		'''
		核心处理程序：检测地标、定位灯光、寻找袋子

		需要验证的问题，当钩子移动的时候，是不是得实时检测地标位置，好做图像校正；

		袋子的坐标应该是固定不变的，如果袋子坐标变化，则是误差。



		:return:
		'''

		# 定位地标
		dest = LandMarkDetecotr(img=img).position_remark()

		# 定位袋子
		bag_detector = BagDetector(dest)
		bags = bag_detector.location_bag()



		while len([bag for bag in bags if bag.finish_move == False]) > 0:
			# 定位激光灯
			laster_detector = LasterDetector(img=dest)
			laster = laster_detector.location_laster()


			for bag in bags:
				print(
					"################################################################################################")
				print("move bag {NO} to train".format(NO=bag.id))
				print("bag position is ({x},{y})".format(x=bag.x, y=bag.y))
				bag.finish_move = True
				print("命令plc复位")

# if self.hockstatus == HockStatus.POSITION_NEARESTBAG:
# 	try:
# 		locationinfo = self.positionservice.find_nearest_bag()
# 		if locationinfo is None:
# 			return
# 		else:
# 			nearest_bag_position, hockposition = locationinfo
# 			img_distance, real_distance, real_x_distance, real_y_distance = self.positionservice.compute_distance(
# 				nearest_bag_position, hockposition)
# 			mylog_debug("最近的袋子距离钩子:{}公分".format(real_distance))
# 			bagnum = len(self.positionservice.bags)
# 			self.bagnums.append(bagnum)
# 			if len(self.bagnums) > 7:
# 				self.foundbagSignal.emit(max(self.bagnums))
# print("向PLC中写入需要移动的X、Y轴移动距离")
# movex = real_x_distance
# movey = real_y_distance
# if img_distance < 200:
# 	self.dropHockSignal.emit((0, 0, 7))  # 暂时写死，钩子找到最近袋子，向下抛7米吧
# 	self.send_positions.append((0, 0, 7))
# else:
# 	self.positionSignal.emit((movex, movey, 0))  # Z轴不变
# 	self.send_positions.append((movex, movey, 0))
# except NotFoundBagException:
# 	self.finishSignal.emit("not found bag ,maybe finish")
# except NotFoundHockException:
# 	self.finishSignal.emit("not found hock ,maybe finish")

# TODO 图像检测是否钩住袋子
# elif self.hockstatus == HockStatus.DROP_HOCK:
# 	# print("图像检测，钩子向下是否抵达袋子")
# 	self.positionservice.compute_hook_location
# 	# print("图像检测，钩子已经挂住袋子，发出拉取袋子命令")
# 	self.pullHockSignal.emit((0, 0, 7))  # 向上拉袋子
# 	self.send_positions.append((0, 0, 7))
# TODO 图像检测是否拉起袋子
# elif self.hockstatus == HockStatus.PULL_HOCK:
# 	# print("图像检测程序检测是否拉起钩子")
# 	self.positionservice.compute_hook_location
# 	# print("图像检测开始检测传送带区域，准备放下袋子")
# 	self.findConveyerBeltSignal.emit((0, 3, 0))

# TODO 图像检测是否抵达放置区
# elif self.hockstatus == HockStatus.FIND_CONVEYERBELT:
# 	# print("定位传送带位置")
# 	self.findConveyerBeltSignal.emit((0, 3, 0))
# 	distance_place = 0
# 	if distance_place == 0:
# 		self.dropBagSignal.emit((0, 0, 7))
#
# elif self.hockstatus == HockStatus.DROP_BAG:
# 	self.rebackSignal.emit((0, 10, 0))

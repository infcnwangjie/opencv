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


class ProcessThread(QThread):
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

	def __init__(self, video_player, IMGHANDLE=None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = True
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		self._hockstatus = HockStatus.POSITION_NEARESTBAG  # 钩子状态会影响定位程序
		self.bags = []
		self.send_positions = []
		self.landmark_detect = LandMarkDetecotr()
		self.bag_detect = BagDetector()
		self.laster_detect = LasterDetector()

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
			# sleep(1 / 8)
			show = self.IMAGE_HANDLE.read()
			if show is None:
				break
			# frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
			# if frame.ndim == 3:
			# 	show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# elif frame.ndim == 2:
			# 	show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

			if self.work:
				dest, success = self.landmark_detect.position_landmark(show)
				self.laster_detect.location_laster(dest, middle_start=250, middle_end=500)
				if success:
					self.bag_detect.location_bags(dest, success, middle_start=100, middle_end=400)
					self.landmark_detect.draw_grid_lines(dest)
				else:
					self.bag_detect.location_bags(dest, middle_start=400, middle_end=600)

				show = dest.copy()

				if show.ndim == 3:
					show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
				elif show.ndim == 2:
					show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

				showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
				# self.video_player.set_img(show)
				self.video_player.setPixmap(QPixmap.fromImage(showImage))
				self.video_player.setScaledContents(True)

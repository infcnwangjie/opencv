import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.processthread import ProcessThread
from app.core.autowork.plcthread import PlcThread
from app.core.plc.plchandle import PlcHandle
from app.log.logtool import logger
from app.status import HockStatus


class IntelligentProcess(object):
	def __init__(self, IMGHANDLE, img_play, plchandle):
		self.plchandle = plchandle
		self._IMGHANDLE = IMGHANDLE
		self.init_imgplay(img_play)
		self.init_imgdetector_thread()
		self.status_show=None

	@property
	def IMGHANDLE(self):
		return self._IMGHANDLE

	@IMGHANDLE.setter
	def IMGHANDLE(self, value):
		self._IMGHANDLE = value
		self.intelligentthread.IMAGE_HANDLE = value

	def init_imgplay(self, img_play):
		self.img_play = img_play

	# self.img_play.left_button_release_signal.connect(self.set_roi)

	def init_imgdetector_thread(self):
		'''初始化图像处理线程'''
		self.intelligentthread = ProcessThread(IMGHANDLE=self.IMGHANDLE, PLCHANDLE=self.plchandle,
		                                       video_player=self.img_play)

	def quickly_stop_work(self):
		'''
		行车紧急停止
		:return:
		'''
		self.intelligentthread.work = False

		try:
			self.plchandle.ugent_stop()
			if self.intelligentthread.save_video:
				self.intelligentthread.update_savevideo.emit(self.intelligentthread.save_video)
		except Exception as e:
			logger(e.__str__(), "error")

	def switch_power(self):
		'''
		行车梯形图电源
		:return:
		'''

		try:
			self.plchandle.power = not self.plchandle.power
		except Exception as e:
			logger(e.__str__(), "error")

	def resetplc(self):
		'''
		行车复位
		:return:
		'''
		try:
			self.plchandle.reset()
		except Exception as  e:
			logger(e.__str__(), "error")

	def save_video(self):
		'''
		留存
		:return:
		'''
		# self.plchandle.reset()
		self.intelligentthread.save_video = not self.intelligentthread.save_video
		print("录像功能为:{}".format(self.intelligentthread.save_video))

import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.processthread import ProcessThread
from app.core.plc.plchandle import PlcHandle
from app.log.logtool import logger
from app.status import HockStatus


# ------------------------------------------------
# 名称：IntelligentProcess
# 功能：作为SERVICE层使用，控制行车移动，启动行车梯形图电源，行车复位灯等
# 状态：在用，后期重构之后会改动
# 作者：王杰  2020-4-15
# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称：init_imgdetector_thread
	# 功能：初始化视频识别线程
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def init_imgdetector_thread(self):
		self.intelligentthread = ProcessThread(IMGHANDLE=self.IMGHANDLE, PLCHANDLE=self.plchandle,
		                                       video_player=self.img_play)

	# ------------------------------------------------
	# 名称：quickly_stop_work
	# 功能：紧急停止，交互方式手动控制行车紧急停止
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-5-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称：switch_power
	# 功能：行车梯形图电源，交互方式启停行车梯形图电源
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-5-xx
	# ------------------------------------------------
	def switch_power(self):

		try:
			self.plchandle.power = not self.plchandle.power
		except Exception as e:
			logger(e.__str__(), "error")

	# ------------------------------------------------
	# 名称：resetplc
	# 功能：行车复位，手动方式控制行车复位
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def resetplc(self):
		'''
		行车复位
		:return:
		'''
		try:
			self.plchandle.reset()
		except Exception as  e:
			logger(e.__str__(), "error")

	#------------------------------------------------
	# 名称：save_video
	# 功能：录制视频
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def save_video(self):
		# self.plchandle.reset()
		self.intelligentthread.save_video = not self.intelligentthread.save_video
		print("录像功能为:{}".format(self.intelligentthread.save_video))

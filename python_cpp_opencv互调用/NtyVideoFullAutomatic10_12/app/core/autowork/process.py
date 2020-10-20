import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.processhandle import ProcessThread
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
	def __init__(self, IMGHANDLE, img_play, plchandle, error_widget=None, dock_img_player=None):
		self.plchandle = plchandle
		self._IMGHANDLE = IMGHANDLE
		self.init_imgplay(img_play, dock_img_player)
		self.init_imgdetector_thread()
		self.status_show = None
		self.error_widget = error_widget

	@property
	def IMGHANDLE(self):
		return self._IMGHANDLE

	@IMGHANDLE.setter
	def IMGHANDLE(self, value):
		self._IMGHANDLE = value
		self.intelligentthread.IMAGE_HANDLE = value

	def init_imgplay(self, img_play, dock_img_player=None):
		self.img_play = img_play
		self.dock_img_player = dock_img_player

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
		                                       video_player=self.img_play, dock_img_player=self.dock_img_player)

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

			if len(self.intelligentthread.target_bag_position) > 0:
				self.intelligentthread.target_bag_position.clear()
			self.intelligentthread.scan_bag = False
			self.intelligentthread.move_to_bag_x = False
			self.intelligentthread.move_to_bag_y = False
			if self.intelligentthread.save_video:
				self.intelligentthread.update_savevideo.emit(self.intelligentthread.save_video)
		except Exception as e:
			logger(e.__str__(), "error")

	# ------------------------------------------------
	# 名称：re_work
	# 功能：重新开始任务
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-9-xx
	# ------------------------------------------------
	def re_work(self):
		self.intelligentthread.work = True
		try:
			self.intelligentthread.target_bag_position.clear()
			self.intelligentthread.move_to_bag_x = False
			self.intelligentthread.move_to_bag_y = False
			self.intelligentthread.scan_bag = False
		except Exception as e:
			logger(e.__str__(), "error")

	# ------------------------------------------------
	# 名称：move_x_center
	# 功能：将行车置中
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-01
	# ------------------------------------------------
	def move(self, east=0, west=0, south=0, nourth=0, up=0, down=0, up_cargohook=0, down_cargohook=0):
		try:
			self.plchandle.move(east, west, south, nourth, up, down, up_cargohook, down_cargohook)
		except Exception as e:
			logger(e.__str__(), "error")

	# ------------------------------------------------
	# 名称：turnon_lamp
	# 功能：打开照明灯
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-9-25
	# ------------------------------------------------
	def turnon_lamp(self):
		try:
			self.plchandle.turnon_lamp()
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
			self.plchandle.power = True
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

	# ------------------------------------------------
	# clear_plc
	# 功能：行车不复位，仅仅清零
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def clear_plc(self):
		'''
		行车清零
		:return:
		'''
		try:
			self.plchandle.clear_plc()
		except Exception as  e:
			logger(e.__str__(), "error")

	# ------------------------------------------------
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

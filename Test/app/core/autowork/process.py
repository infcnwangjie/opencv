import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.processthread import ProcessThread
from app.core.autowork.plcthread import PlcThread
from app.core.processers.landmark_detector import LandMarkDetecotr
from app.core.plc.plchandle import PlcHandle
from app.core.beans.locationservice import PointLocationService
from app.core.processers.bag_detector import BagDetector
from app.core.processers.laster_detector import LasterDetector
from app.status import HockStatus


class IntelligentProcess(object):
	def __init__(self, IMGHANDLE, img_play, plc_status_show):
		self.plchandle = PlcHandle()
		self._IMGHANDLE = IMGHANDLE
		self.init_imgplay(img_play)
		self.plc_status_show = plc_status_show
		self.init_imgdetector_thread()
		self.check_plc_status()

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

	def check_plc_status(self):
		'''检测plc状态'''
		self.plc_status_show.setText('开启' if self.plchandle.is_open() else "关闭")



	def init_imgdetector_thread(self):
		'''初始化图像处理线程'''
		self.intelligentthread = ProcessThread(IMGHANDLE=self.IMGHANDLE,
		                                           video_player=self.img_play)
		# self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG
		# self.intelligentthread.positionSignal.connect(self.writetoplc_imgsignal_process)  # 发送移动位置
		# self.intelligentthread.dropHockSignal.connect(self.drophock_imgsignal_process)  # 命令放下钩子
		# self.intelligentthread.pullHockSignal.connect(self.pullhock_imgsignal_process)  # 命令拉起钩子
		# self.intelligentthread.findConveyerBeltSignal.connect(self.findconveyerbelt_imgsignal_process)  # 命令移动袋子到传送带
		# self.intelligentthread.dropBagSignal.connect(self.drop_bag_imgsignal_process)
		# self.intelligentthread.rebackSignal.connect(self.reback_imgsignal_process)
		# self.intelligentthread.finishSignal.connect(self.finish_imgsignal_process)
		# self.intelligentthread.foundbagSignal.connect(self.editbagnum_imgsignal_process)


	def quickly_stop_work(self):
		self.plchandle.ugent_stop()



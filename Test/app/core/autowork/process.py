import cv2
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.intelligentthread import IntelligentThread
from app.core.autowork.plcthread import PlcThread
from app.core.processers.landmark_detector import LandMarkDetecotr
from app.core.plc.plchandle import PlcHandle
from app.core.location.locationservice import PointLocationService
from app.core.processers.bag_detector import BagDetector
from app.core.processers.laster_detector import LasterDetector
from app.status import HockStatus


class IntelligentProcess(object):
	def __init__(self, IMGHANDLE, img_play, plc_status_show):
		self.plchandle = PlcHandle()
		self._IMGHANDLE = IMGHANDLE
		self.init_imgplay(img_play)
		self.plc_status_show = plc_status_show
		self.init_plc_thread()
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
		self.plc_status_show.setText('plc连接成功' if self.plchandle.status else "很抱歉，连接失败")

	def init_plc_thread(self):
		'''初始化plc线程'''
		self.plcthread = PlcThread(plchandle=self.plchandle)
		self.plcthread.askforSingnal.connect(self.askposition_plcsignal_process)
		self.plcthread.moveSignal.connect(self.notneedposition_plcsignal_process)

	def init_imgdetector_thread(self):
		'''初始化图像处理线程'''
		self.intelligentthread = IntelligentThread(IMGHANDLE=self.IMGHANDLE, positionservice=PointLocationService(),
		                                           video_player=self.img_play)
		self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG
		self.intelligentthread.positionSignal.connect(self.writetoplc_imgsignal_process)  # 发送移动位置
		self.intelligentthread.dropHockSignal.connect(self.drophock_imgsignal_process)  # 命令放下钩子
		self.intelligentthread.pullHockSignal.connect(self.pullhock_imgsignal_process)  # 命令拉起钩子
		self.intelligentthread.findConveyerBeltSignal.connect(self.findconveyerbelt_imgsignal_process)  # 命令移动袋子到传送带
		self.intelligentthread.dropBagSignal.connect(self.drop_bag_imgsignal_process)
		self.intelligentthread.rebackSignal.connect(self.reback_imgsignal_process)
		self.intelligentthread.finishSignal.connect(self.finish_imgsignal_process)
		self.intelligentthread.foundbagSignal.connect(self.editbagnum_imgsignal_process)

	def askposition_plcsignal_process(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def notneedposition_plcsignal_process(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def editbagnum_imgsignal_process(self, bagnum):
		self.bag_num_show.setText(str(bagnum))
		print("bag num is {}".format(bagnum))

	def writetoplc_imgsignal_process(self, position):
		'''接收反馈信号'''
		x, y, z = position
		# print("X轴移动：{}，Y轴移动{},z轴移动{}".format(*position))
		# print(position)
		self.plchandle.write_position(position)
		self.plcthread.work = True
		self.intelligentthread.work = True

	def drophock_imgsignal_process(self, position):
		_x, _y, z = position
		print("已经定位袋子，正在放下钩子,z:{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.DROP_HOCK
		self.currentstatus_edit.setText("已经定位袋子，正在放下钩子")

	def pullhock_imgsignal_process(self, position):
		_x, _y, z = position
		self.currentstatus_edit.setText("正在拉起袋子,向上拉取{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.PULL_HOCK

	def findconveyerbelt_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("袋子要到传送带，先向x,y,z各自移动{},{},{}米".format(*position))
		self.intelligentthread.hockstatus = HockStatus.FIND_CONVEYERBELT

	def drop_bag_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("移动袋子抵达传送带，向下移动钩子{}米,放下袋子".format(position[2]))
		self.intelligentthread.hockstatus = HockStatus.DROP_BAG

	def reback_imgsignal_process(self, position):
		self.plchandle.write_position(position)
		print("放下钩子")
		self.currentstatus_edit.setText("复位")
		self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG

	def finish_imgsignal_process(self, info):
		# print("工作完成")
		# print(info)
		self.plchandle.ugent_stop()
		self.plcthread.work = False

	def afterreback(self, info):
		'''处理重置'''
		print(info)
		self.plcthread.work = True
		self.intelligentthread.work = False

	def monitor_all_once(self, image=None):
		if image:
			img = cv2.imread(image)
		else:
			img = cv2.imread('d:/2020-04-10-15-26-22test.bmp')

		# 定位地标
		dest = LandMarkDetecotr(img=img).position_remark()

		# 定位袋子
		bag_detector = BagDetector(dest)
		bags = bag_detector.location_bag()

		# 定位激光灯
		laster_detector = LasterDetector(img=dest)
		print(laster_detector.location_laster())

		while len([bag for bag in bags if bag.finish_move == False]) > 0:
			for bag in bags:
				print(
					"################################################################################################")
				print("move bag {NO} to train".format(NO=bag.id))
				print("bag position is ({x},{y})".format(x=bag.x, y=bag.y))
				bag.finish_move = True
				print("命令plc复位")

		show = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)
		showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
		self.img_play.setPixmap(QPixmap.fromImage(showImage))
		self.img_play.setScaledContents(True)

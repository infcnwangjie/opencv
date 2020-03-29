from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from app.core.exceptions.sdkexceptions import SdkException
from app.core.plc.plchandle import PlcHandle
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK


class IntelligentThread(QThread):
	finishSignal = pyqtSignal(str)
	positionSignal = pyqtSignal(tuple)

	def __init__(self, video_player, IMGHANDLE=None,
	             positionservice: PointLocationService = None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = False
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		self.positionservice = positionservice  # 指令处理器

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

		while self.play:
			sleep(1 / 22)
			frame = self.IMAGE_HANDLE.read()
			if frame is None:
				self.sinOut.emit("frame is None！")
				self.finish = True
				self.sinOut.emit("操作完毕")
				break
			frame = cv2.resize(frame, (800, 800))

			if frame.ndim == 3:
				show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			elif frame.ndim == 2:
				show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

			if self.work:
				self.positionservice.img = show
				self.process()
				show = self.positionservice.img

			showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(showImage))
			self.video_player.setScaledContents(True)

	def process(self):
		location_info = self.positionservice.computelocations()
		if location_info is None:
			return None
		img_distance, real_distance, real_x_distance, real_y_distance = location_info
		self.nearest_bag = self.positionservice.nearestbag
		self.hock = self.positionservice.hock
		# 运行定位逻辑,并实时更新钩子与选中袋子实时位置
		# print("向PLC中写入需要移动的X、Y轴移动距离")
		movex = real_x_distance
		movey = real_y_distance
		# print("X轴移动：{}，Y轴移动{}".format(real_x_distance, real_y_distance))
		self.positionSignal.emit((movex, movey))

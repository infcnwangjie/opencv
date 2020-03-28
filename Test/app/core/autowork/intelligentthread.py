from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from app.core.autowork.instruct import InstructSender
from app.core.exceptions.sdkexceptions import SdkException
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK


class IntelligentThread(QThread):
	sinOut = pyqtSignal(str)

	def __init__(self, video_player, IMGHANDLE=None, instruct_sender: InstructSender = None, parent=None):
		super().__init__(parent=parent)
		self.working = False
		self.playing = True
		self.finish = False
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		self.instruct_sender = instruct_sender  # 指令处理器

	def __del__(self):
		self.working = False
		if self.IMAGE_HANDLE:
			try:
				self.IMAGE_HANDLE.release()
			except SdkException as e:
				raise e

	def run(self):

		while self.playing:
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

			if self.working:
				# self.sinOut.emit("开启智能工作！")
				with PointLocationService(img=show, print_or_no=False) as  a:
					a.computelocations(flag=BAG_AND_LANDMARK)
					show = a.img
			showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(showImage))
			self.video_player.setScaledContents(True)

	def stopwork(self):
		'''关闭智能工作'''
		self.working = False

	def autowork(self):
		self.working = True

	def stopcamera(self):
		'''关闭摄像机'''
		self.playing = False

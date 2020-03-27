from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class WorkThread(QThread):
	sinOut = pyqtSignal(str)

	def __init__(self, video_player, video_file=None, parent=None):
		super().__init__(parent=parent)
		self.working = False
		self.playing = True
		self.finish = False
		self.video_player = video_player
		self.video_file = video_file
		if self.video_file:
			self.cap = cv2.VideoCapture(self.video_file)
		else:
			self.cap = cv2.VideoCapture(0)

	def __del__(self):
		self.working = False
		self.cap.release()

	def run(self):
		if self.finish == True:
			print("读取完毕重新开始")
			if self.video_file:
				self.cap = cv2.VideoCapture(self.video_file)
			else:
				self.cap = cv2.VideoCapture(0)

		while self.playing:
			sleep(1 / 23)
			ret, frame = self.cap.read()
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

			showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)


			# self.label.setPixmap(QPixmap.fromImage(showImage))

			# temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(showImage))
			self.video_player.setScaledContents(True)
			# if self.working:
			# 	self.sinOut.emit("开启智能工作！")
			# else:
			# 	self.sinOut.emit("关闭智能工作！")



	def stopwork(self):
		'''关闭智能工作'''
		self.working = False

	def stopcamera(self):
		'''关闭摄像机'''
		self.playing = False

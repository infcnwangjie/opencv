# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal

from app.core.plc.plchandle import PlcHandle
from app.status import HockStatus


class PlcThread(QThread):
	'''
	钩子运动的时候，一定要向PLC中写入X，Y位置参数

	'''
	askforSingnal = pyqtSignal(str)
	moveSignal = pyqtSignal(str)

	def __init__(self, plchandle: PlcHandle):
		super().__init__(parent=None)
		self.plchandle = plchandle

	def run(self):
		if self.work == True:
			move_status = self.plchandle.read_status()
			if move_status == 0:
				#静止态，需要索要定位信息
				self.askforSingnal.emit("plc need hock  position!")
			elif move_status == 1:
				# 运动态不需要定位信息
				self.moveSignal.emit("hock is moveing ,dont need position")

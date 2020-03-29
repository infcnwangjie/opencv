from PyQt5.QtCore import QThread, pyqtSignal

from app.core.autowork.instruct import InstructSender
from app.status import HockStatus


class HockThread(QThread):
	'''钩子运动的时候，一定要向PLC中写入X，Y位置参数'''
	sinOut = pyqtSignal(str)

	def __init__(self, instruct_sender: InstructSender = None, status=HockStatus.REBACK):
		self.instruct_sender = instruct_sender  # 指令处理器
		self.status = status

	def run(self):
		moveposition = self.instruct_sender.get_moveposition_hock()

		if moveposition is None:
			# 如果没有获取指令，钩子线程会发射信号给控制中心,并且更新钩子状态
			# self.sinOut.emit()
			self.status = HockStatus.STOP

		else:
			self.instruct_sender.move()

# encoding:utf-8
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from app.views.main_window import CenterWindow, MainWindow

if __name__ == '__main__':
	"""
	今日工作:修改程序，修改定位精度逻辑；配合朱工、胡工联调测试；

	明日工作：
	减少识别迭代次数，迭代调整增加休眠等待，联调测试
	"""
	app = QApplication(sys.argv)
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

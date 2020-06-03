# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

from app.views.main_window import CenterWindow, MainWindow


if __name__ == '__main__':
	'''
	如果检测不到地标,说明地标ROI需要更换了
	'''
	app = QApplication(sys.argv)
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

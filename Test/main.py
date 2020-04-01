# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.views.main_window import CenterWindow, MainWindow

if __name__ == '__main__':
	app = QApplication(sys.argv)  # 生成应用
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

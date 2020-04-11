# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

from app.core.location.locationservice import PointLocationService, BAG_AND_LANDMARK
from app.views.main_window import CenterWindow, MainWindow

if __name__ == '__main__':
	app = QApplication(sys.argv)
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

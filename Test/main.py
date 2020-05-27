# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from app.views.main_window import CenterWindow, MainWindow

'''
today to do:
	钩子位置还没有定位，只是用激光灯定位了钩子的Y，钩子的X轴需要识别钩子
 １、视频识别软件，袋子检测位置去重与排除异常袋子
 2、视频识别软件，加入实时纠偏
 3、视频识别软件，加入是否越界判断
'''

if __name__ == '__main__':
	app = QApplication(sys.argv)
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

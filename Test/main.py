# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

from app.views.main_window import CenterWindow, MainWindow

'''
#TODO 5.30 
1、如果视频识别看不到地标怎么处理   向前移动一米
2、如果视频软件看不到目标怎么处理   向前移动一米
3、如果视频中袋子数目是0，怎么处理
4、怎么真正防止行车越界造成危险
'''

if __name__ == '__main__':
	app = QApplication(sys.argv)
	mainwindow = MainWindow()
	mainwindow.show()
	sys.exit(app.exec_())

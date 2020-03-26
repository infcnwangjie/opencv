# encoding:utf-8
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow

from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.views.main_window import MainWindowUi, CenterWindow

if __name__ == '__main__':
	app = QApplication(sys.argv)  # 生成应用
	mainwindow=CenterWindow()
	mainwindow.show()

	sys.exit(app.exec_())
# app = QApplication(sys.argv)
# im = cv2.imread('C:/work/imgs/test/bag6.bmp')
# with PointLocationService(img=im,print_or_no=True) as  a:
# 	a.computelocations(flag=BAG_AND_LANDMARK)
# 	a.move()

# window = Login_Window()
# center_window = CenterWindowUi()
# window.show()
# sys.exit(app.exec_())

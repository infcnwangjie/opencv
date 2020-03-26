# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\能特异代码库\uis\videowindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget

from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK


class MainWindowUi(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1289, 1000)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("智能行车")

		# vbox_layout=QtWidgets.QVBoxLayout()
		hbox_layout = QtWidgets.QHBoxLayout()

		self.operatorBox = QtWidgets.QGroupBox(self.centralwidget)
		self.operatorBox.setGeometry(QtCore.QRect(20, 40, 221, 800))
		self.operatorBox.setObjectName("operatorBox")
		self.only_start_button = QtWidgets.QToolButton(self.operatorBox)
		self.only_start_button.setGeometry(QtCore.QRect(10, 30, 171, 41))
		self.only_start_button.setObjectName("only_start_button")
		self.stop_button = QtWidgets.QToolButton(self.operatorBox)
		self.stop_button.setGeometry(QtCore.QRect(10, 160, 171, 41))
		self.stop_button.setObjectName("stop_button")
		self.start_and_store_button = QtWidgets.QToolButton(self.operatorBox)
		self.start_and_store_button.setGeometry(QtCore.QRect(10, 90, 171, 41))
		self.start_and_store_button.setObjectName("start_and_store_button")

		hbox_layout.addWidget(self.operatorBox)

		self.videoBox = QtWidgets.QGroupBox(self.centralwidget)
		self.videoBox.setGeometry(QRect(250, 40, 1000, 800))
		# self.videoBox.setMaximumSize(QSize(1000, 800))
		# self.videoBox.resize(QSize(1100,850))
		self.videoBox.setObjectName("videoBox")
		self.picturelabel = QtWidgets.QLabel(self.videoBox)
		# self.picturelabel.setGeometry(QtCore.QRect(50, 40, 900, 900))
		self.picturelabel.setText("")
		self.picturelabel.setObjectName("picturelabel")
		# self.picturelabel.adjustSize()
		# self.picturelabel.resize(QSize(1000,800))
		hbox_layout.addWidget(self.videoBox)

		# vbox_layout.addLayout(hbox_layout)

		# self.videosBox = QtWidgets.QGroupBox(self.centralwidget)
		# # self.videosBox.setGeometry(QtCore.QRect(20, 870, 1191, 165))
		# # self.videosBox.setMaximumSize(QtCore.QSize(1191, 16777215))
		# self.videosBox.resize(QSize(1000,300))
		# self.videosBox.setObjectName("videosBox")

		# vbox_layout.addWidget(self.videosBox)

		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1289, 26))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		MainWindow.setLayout(hbox_layout)
		self.retranslateUi(MainWindow)

		# MainWindow.showMaximized()
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
		self.only_start_button.setText(_translate("MainWindow", "仅开始"))
		self.stop_button.setText(_translate("MainWindow", "结束"))
		self.start_and_store_button.setText(_translate("MainWindow", "开始并录制"))
		# self.videosBox.setTitle(_translate("MainWindow", "回放录像"))
		self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QMainWindow, MainWindowUi):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.only_start_button.clicked.connect(self.start)
		self.start_and_store_button.clicked.connect(self.start_and_store)
		self.stop_button.clicked.connect(self.stop)

	def changeEvent(self, e):
		if e.type() == QtCore.QEvent.WindowStateChange:
			if self.isMinimized():
				print("窗口最小化")
			elif self.isMaximized():
				print("窗口最大化")
				desktop = QDesktopWidget()
				screen_width = desktop.screenGeometry().width()
				screen_height = desktop.screenGeometry().height()
				print(screen_width,screen_height)
				self.picturelabel.resize(QSize(screen_width*0.8-20,screen_height*0.8-10))
				self.videoBox.resize(QSize(screen_width*0.8,screen_height*0.8))
				self.operatorBox.resize(QSize(self.operatorBox.width(),screen_height*0.8))
			elif self.isFullScreen():
				print("全屏显示")
			elif self.isActiveWindow():
				print("活动窗口")
		QtWidgets.QWidget.changeEvent(self, e)

	def start(self):
		img = cv2.imread('C:/work/imgs/test/bag6.bmp')

		with PointLocationService(img=img, print_or_no=False) as  a:
			a.computelocations(flag=BAG_AND_LANDMARK)
			img=a.move()
		show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
		self.picturelabel.setPixmap(QPixmap.fromImage(showImage))
		self.picturelabel.setScaledContents(True)

		print("start")

	def start_and_store(self):
		print("start_and_store")

	def stop(self):
		print("stop")

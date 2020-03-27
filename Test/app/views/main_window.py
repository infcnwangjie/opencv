# -*- coding: utf-8 -*-
import os

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, QRect, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget

from app.config import ICON_DIR
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.core.video.camero import CameroThread


class MainWindowUi(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1289, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("智能行车")

        hbox_layout = QtWidgets.QHBoxLayout()

        self.operatorBox = QtWidgets.QGroupBox(self.centralwidget)
        self.operatorBox.setGeometry(QtCore.QRect(20, 40, 221, 800))
        self.operatorBox.setObjectName("operatorBox")
        self.open_camera_button = QtWidgets.QPushButton(self.operatorBox)
        self.open_camera_button.setGeometry(QtCore.QRect(10, 30, 171, 41))
        self.open_camera_button.setIcon(QIcon(os.path.join(ICON_DIR, "camera.png")))
        self.open_camera_button.setObjectName("open_camera_button")

        self.auto_work_button = QtWidgets.QPushButton(self.operatorBox)
        self.auto_work_button.setGeometry(QtCore.QRect(10, 90, 171, 41))
        self.auto_work_button.setIcon(QIcon(os.path.join(ICON_DIR, "auto.png")))
        self.auto_work_button.setObjectName("auto_work_button")

        self.stop_button = QtWidgets.QPushButton(self.operatorBox)
        self.stop_button.setGeometry(QtCore.QRect(10, 160, 171, 41))
        self.stop_button.setIcon(QIcon(os.path.join(ICON_DIR, "stop.png")))
        self.stop_button.setObjectName("stop_button")

        self.quit_button = QtWidgets.QPushButton(self.operatorBox)
        self.quit_button.setGeometry(QtCore.QRect(10, 230, 171, 41))
        self.quit_button.setIcon(QIcon(os.path.join(ICON_DIR, "quit.png")))
        self.quit_button.setObjectName("quit_button")

        hbox_layout.addWidget(self.operatorBox)

        self.videoBox = QtWidgets.QGroupBox(self.centralwidget)
        self.videoBox.setGeometry(QRect(250, 40, 1000, 800))
        self.videoBox.setObjectName("videoBox")
        self.picturelabel = QtWidgets.QLabel(self.videoBox)
        self.picturelabel.setText("")
        self.picturelabel.setObjectName("picturelabel")
        hbox_layout.addWidget(self.videoBox)

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
        MainWindow.setWindowTitle(_translate("MainWindow", "视频识别机械手"))
        self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
        self.open_camera_button.setText(_translate("MainWindow", "开启摄像头"))
        self.stop_button.setText(_translate("MainWindow", "结束"))
        self.auto_work_button.setText(_translate("MainWindow", "智能工作"))
        self.quit_button.setText(_translate("MainWindow", "退出系统"))
        self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QMainWindow, MainWindowUi):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.open_camera_button.clicked.connect(self.opencamera)
        self.auto_work_button.clicked.connect(self.autowork)
        self.stop_button.clicked.connect(self.stop)
        self.quit_button.clicked.connect(QCoreApplication.quit)

        self.thread = CameroThread(video_file="E:\\小兔子视频\\building.mp4",video_player=self.picturelabel)
        self.thread.sinOut.connect(self.info)

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.WindowStateChange:
            if self.isMinimized():
                print("窗口最小化")
            elif self.isMaximized():
                print("窗口最大化")
                desktop = QDesktopWidget()
                screen_width = desktop.screenGeometry().width()
                screen_height = desktop.screenGeometry().height()
                print(screen_width, screen_height)
                self.picturelabel.resize(QSize(screen_width * 0.7 - 20, screen_height * 0.8))
                self.operatorBox.resize(QSize(self.operatorBox.width(), screen_height * 0.8))
                self.videoBox.resize(QSize(screen_width * 0.7, screen_height * 0.8))

            elif self.isFullScreen():
                print("全屏显示")
            elif self.isActiveWindow():
                print("活动窗口")
        QtWidgets.QWidget.changeEvent(self, e)

    def opencamera(self):
        img = cv2.imread('C:/work/imgs/test/bag6.bmp')
        with PointLocationService(img=img, print_or_no=False) as  a:
            a.computelocations(flag=BAG_AND_LANDMARK)
            img = a.move()
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.picturelabel.setPixmap(QPixmap.fromImage(showImage))
        self.picturelabel.setScaledContents(True)


    def autowork(self):
        self.thread.start()

    def stop(self):
        self.thread.stop()

    def info(self,infomessage):
        print(infomessage)

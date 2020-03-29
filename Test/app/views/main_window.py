# -*- coding: utf-8 -*-
import itertools
import os

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
    QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit

from app.config import ICON_DIR
from app.core.autowork.hockthread import HockThread
from app.core.plc.plchandle import PlcHandle
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.core.video.imageprovider import ImageProvider
from app.core.autowork.intelligentthread import IntelligentThread
from app.status import HockStatus


class CentWindowUi(object):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1289, 1000)
        all_layout = QtWidgets.QHBoxLayout()

        self.storedbox = QtWidgets.QGroupBox(self)
        self.storedbox.setObjectName("storedbox")
        self.storedbox.setTitle("已存录像")

        layout = QtWidgets.QVBoxLayout()

        # days = ['2020-03-26', '2020-03-27']
        videos = ['2020-03-26 08:40:21.mp4', '2020-03-26 09:41:21.mp4', '2020-03-26 14:21:21.mp4',
                  '2020-03-26 15:21:21.mp4', '2020-03-27 09:34:21.mp4', '2020-03-27 11:50:34.mp4',
                  '2020-03-27 14:40:34.mp4']
        groupinfo = itertools.groupby(videos, key=lambda videofile: videofile[0:10])
        self.tree = QTreeWidget()

        self.tree.setHeaderLabels(['视频录像'])
        self.tree.setColumnCount(1)
        self.tree.setColumnWidth(0, 180)
        for datestr, files in groupinfo:
            root = QTreeWidgetItem(self.tree)
            root.setText(0, datestr)
            root.setIcon(0, QIcon(os.path.join(ICON_DIR, "catalogue.png")))
            for filepath in files:
                child = QTreeWidgetItem(root)
                child.setText(0, filepath)
                # child1.setText(1, 'ios')
                child.setIcon(0, QIcon(os.path.join(ICON_DIR, 'autowork.png')))
                # child1.setCheckState(0, Qt.Checked)
                root.addChild(child)

        self.tree.clicked.connect(self.onClicked)
        self.tree.expandAll()
        layout.addWidget(self.tree)
        self.storedbox.setLayout(layout)

        all_layout.addWidget(self.storedbox)

        self.videoBox = QtWidgets.QGroupBox(self)
        self.videoBox.setObjectName("videoBox")
        all_layout.addWidget(self.videoBox)

        self.picturelabel = QtWidgets.QLabel(self)
        self.picturelabel.setObjectName("picturelabel")
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.addWidget(self.picturelabel)
        self.videoBox.setLayout(video_layout)

        # 右侧按钮操作区域
        self.operatorBox = QtWidgets.QGroupBox(self)
        self.operatorBox.setObjectName("operatorBox")
        all_layout.addWidget(self.operatorBox)

        self.play_button = QtWidgets.QToolButton(self)
        self.play_button.setIcon(QIcon(os.path.join(ICON_DIR, "play.png")))
        self.play_button.setIconSize(QSize(60, 60))
        self.play_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.play_button.setObjectName("play_button")
        self.play_button.setStyleSheet("border:none")

        operate_layout = QtWidgets.QGridLayout()
        operate_layout.addWidget(self.play_button, *(0, 0))

        self.stop_button = QtWidgets.QToolButton(self)
        self.stop_button.setIcon(QIcon(os.path.join(ICON_DIR, "stop.png")))
        self.stop_button.setIconSize(QSize(60, 60))
        self.stop_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setStyleSheet("border:none")
        operate_layout.addWidget(self.stop_button, *(0, 1))

        baginfo_layout = QtWidgets.QFormLayout()
        bagnum_label = QLabel("袋子总数：")
        self.bagnum_edit = QLineEdit()
        baginfo_layout.addRow(bagnum_label, self.bagnum_edit)
        restbaglabel = QLabel("剩余袋子：")
        self.restbagnum_edit = QLineEdit()
        baginfo_layout.addRow(restbaglabel, self.restbagnum_edit)

        # 添加袋子信息
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addLayout(operate_layout)
        right_layout.addLayout(baginfo_layout)

        self.operatorBox.setLayout(right_layout)

        self.videoBox.setStyleSheet('''QLabel{color:black}
		                                QLabel{background-color:lightgreen}
		                                QLabel{border:2px}
		                                QLabel{border-radius:10px}
		                             QLabel{padding:2px 4px}''')

        all_layout.setStretch(0, 2)
        all_layout.setStretch(1, 6)
        all_layout.setStretch(2, 2)
        self.setLayout(all_layout)
        self.retranslateUi(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("MainWindow", "视频识别机械手"))
        self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
        self.play_button.setText(_translate("MainWindow", "开始"))
        self.stop_button.setText(_translate("MainWindow", "停止"))
        self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QWidget, CentWindowUi):
    def __init__(self, IMGHANDLE=None):
        super().__init__()
        self.setupUi(self)
        self.play_button.clicked.connect(self.play)
        self.stop_button.clicked.connect(self.stop)
        self.positionservice = PointLocationService()
        self.plchandle = PlcHandle()
        self.intelligentthread = IntelligentThread(IMGHANDLE=IMGHANDLE, positionservice=self.positionservice,
                                                   video_player=self.picturelabel)
        self.hockthread = HockThread(plchandle=self.plchandle)
        self.hockthread.askforSingnal.connect(self.askforposition)
        self.hockthread.moveSignal.connect(self.stoplocation_shorttime)
        self.intelligentthread.finishSignal.connect(self.finish)
        self.intelligentthread.positionSignal.connect(self.writeposition)
        self.intelligentthread.rebackSignal.connect(self.afterreback)
        self.intelligentthread.foundbagSignal.connect(self.editbagnum)

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

            elif self.isFullScreen():
                print("全屏显示")
            elif self.isActiveWindow():
                print("活动窗口")
        QtWidgets.QWidget.changeEvent(self, e)

    def onClicked(self, qmodeLindex):
        item = self.tree.currentItem()
        print('Key=%s,value=%s' % (item.text(0), item.text(1)))

    def play(self):
        if self.intelligentthread.IMAGE_HANDLE:
            self.intelligentthread.play = True
            self.intelligentthread.start()
            self.hockthread.on = True
            self.hockthread.start()
        else:
            QMessageBox.warning(self, "警告",
                                self.tr("还没有开启摄像头或者选择播放视频!"))
            print("关闭")

    def autowork(self):
        self.intelligentthread.work = True

    def stop(self):
        '''暂停摄像机'''
        print("关闭摄像")
        self.intelligentthread.play = False

    def finish(self, info):
        # print("工作完成")
        # print(info)
        pass

    def askforposition(self, info: str):
        print(info)
        self.intelligentthread.work = True
        self.hockthread.work = False

    def stoplocation_shorttime(self, info: str):
        print(info)
        self.intelligentthread.work = False
        self.hockthread.work = True

    def editbagnum(self,bagnum):
        self.bagnum_edit.setText(str(bagnum))
        print( "bag num is {}".format(bagnum))

    def writeposition(self, position):
        '''接收反馈信号'''
        x, y, z = position
        # print("X轴移动：{}，Y轴移动{},z轴移动{}".format(*position))
        # print(position)
        self.plchandle.write_position(position)
        if z < 0:
            self.intelligentthread.hockstatus = HockStatus.DROP_HOCK
            print("已经定位袋子，正在放下钩子")
        elif z > 0:
            self.intelligentthread.hockstatus = HockStatus.DROP_BAG
        self.hockthread.work = False
        self.intelligentthread.work = True

    def afterreback(self, info):
        '''处理重置'''
        print(info)
        self.hockthread.work = True
        self.intelligentthread.work = False

    def test(self):
        img = cv2.imread('C:/work/imgs/test/bag6.bmp')
        with PointLocationService(img=img, print_or_no=False) as  a:
            a.computelocations(flag=BAG_AND_LANDMARK)
            img = a.move()
        img = cv2.resize(img, (800, 800))
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.picturelabel.setPixmap(QPixmap.fromImage(showImage))
        self.picturelabel.setScaledContents(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1289, 1000)
        self.centralwidget = CenterWindow()  # 创建一个文本编辑框组件
        self.setCentralWidget(self.centralwidget)  # 将它设置成QMainWindow的中心组件。中心组件占据了所有剩下的空间。
        self.menu_toolbar_ui()

    def menu_toolbar_ui(self):
        openFileAction = QAction(QIcon(os.path.join(ICON_DIR, 'openfile.png')), '打开', self)
        openFileAction.setShortcut('Ctrl+F')
        openFileAction.setStatusTip('打开文件')
        openFileAction.triggered.connect(self.openfile)

        exitAction = QAction(QIcon(os.path.join(ICON_DIR, 'quit.png')), '退出', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('退出应用')
        exitAction.triggered.connect(self.close)

        openCameraAction = QAction(QIcon(os.path.join(ICON_DIR, 'camera.png')), '摄像头', self)
        openCameraAction.setShortcut('Ctrl+o')
        openCameraAction.setStatusTip('打开摄像头')
        openCameraAction.triggered.connect(self.openCamera)

        stopCameraAction = QAction(QIcon(os.path.join(ICON_DIR, 'close.png')), '关闭摄像头', self)
        stopCameraAction.setShortcut('Ctrl+q')
        stopCameraAction.setStatusTip('关闭摄像头')
        stopCameraAction.triggered.connect(self.stopCamera)

        robotAction = QAction(QIcon(os.path.join(ICON_DIR, 'robot.png')), '自动抓取模式', self)
        robotAction.setShortcut('Ctrl+o')
        robotAction.setStatusTip('自动抓取模式')
        robotAction.triggered.connect(self.work_as_robot)

        testAction = QAction(QIcon(os.path.join(ICON_DIR, 'test.png')), '测试模式', self)
        testAction.setShortcut('Ctrl+t')
        testAction.setStatusTip('测试模式')
        testAction.triggered.connect(self.test)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&文件')
        fileMenu.addAction(openFileAction)
        fileMenu.addAction(exitAction)

        cameraMenu = menubar.addMenu('&摄像头')
        cameraMenu.addAction(openCameraAction)
        cameraMenu.addAction(stopCameraAction)

        openFileToolBar = self.addToolBar('OpenFile')
        openFileToolBar.addAction(openFileAction)

        exitToolbar = self.addToolBar('Exit')
        exitToolbar.addAction(exitAction)

        openCameraToolbar = self.addToolBar("OpenCamera")
        openCameraToolbar.addAction(openCameraAction)

        closeToolbar = self.addToolBar("CloseCamera")
        closeToolbar.addAction(stopCameraAction)

        intellectToolbar = self.addToolBar("Intellect")
        intellectToolbar.addAction(robotAction)

        testToolbar = self.addToolBar("Test")
        testToolbar.addAction(testAction)

        self.setWindowTitle('Main window')
        self.statusBar().show()

    # self.show()

    def openCamera(self):
        # 正常情况读取sdk
        imagehandle = ImageProvider(ifsdk=True)
        self.centralwidget.intelligentthread.IMAGE_HANDLE = imagehandle
        self.centralwidget.play()
        self.statusBar().showMessage("已经开启摄像头!", 0)

    # self.statusBar().show()

    def stopCamera(self):
        '''关闭摄像机'''
        del self.centralwidget.intelligentthread.IMAGE_HANDLE
        self.statusBar().showMessage("已经关闭摄像头!", 0)

    # self.statusBar().show()

    def work_as_robot(self):
        '''开始智能抓取'''
        self.centralwidget.autowork()
        self.statusBar().showMessage("已经开启智能识别!", 0)

    # self.statusBar().show()

    def openfile(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "选取文件",
                                                         "./",
                                                         "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        if filename and os.path.isfile(filename) and os.path.exists(filename):
            imagehandle = ImageProvider(videofile=filename, ifsdk=False)
        else:
            # 正常情况读取sdk
            imagehandle = ImageProvider(ifsdk=True)
        self.centralwidget.intelligentthread.IMAGE_HANDLE = imagehandle
        self.centralwidget.play()
        print(filename, filetype)

    def test(self):
        self.centralwidget.test()

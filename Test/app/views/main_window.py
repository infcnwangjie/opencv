# -*- coding: utf-8 -*-
import itertools
import os
import re

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit, QSplitter

from app.config import ICON_DIR, SDK_OPEN, DEBUG
from app.core.autowork.plcthread import PlcThread
from app.core.plc.plchandle import PlcHandle
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.core.video.imageprovider import ImageProvider
from app.core.autowork.intelligentthread import IntelligentThread
from app.status import HockStatus
import app.icons.resource

class CentWindowUi(object):
	movie_pattern = re.compile("\d{4}-\d{2}-\d{2}-\d{2}.*")

	def setupUi(self, Form):
		Form.setObjectName("Form")
		Form.resize(1289, 1000)
		all_layout = QtWidgets.QHBoxLayout()

		self.storedbox = QtWidgets.QGroupBox(self)
		self.storedbox.setObjectName("storedbox")
		self.storedbox.setTitle("已存录像")

		layout = QtWidgets.QVBoxLayout()

		# days = ['2020-03-26', '2020-03-27']
		videos = []
		for file in os.listdir("D:/video"):
			# if os.path.isfile(file):
			matchresult = re.match(self.movie_pattern, file)
			# print(file)
			if matchresult:
				videos.append(matchresult.group(0))

		groupinfo = itertools.groupby(videos, key=lambda videofile: videofile[0:10])
		self.tree = QTreeWidget()

		self.tree.setHeaderLabels(['视频录像'])
		self.tree.setColumnCount(1)
		self.tree.setColumnWidth(0, 180)
		for datestr, files in groupinfo:
			root = QTreeWidgetItem(self.tree)
			root.setText(0, datestr)
			root.setIcon(0, QIcon(":icons/catalogue.png"))
			for filepath in files:
				child = QTreeWidgetItem(root)
				child.setText(0, filepath)
				# child1.setText(1, 'ios')
				child.setIcon(0, QIcon(":icons/video.png"))
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
		self.play_button.setIcon(QIcon(":icons/play.png"))
		self.play_button.setIconSize(QSize(60, 60))
		self.play_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.play_button.setObjectName("play_button")
		self.play_button.setStyleSheet("border:none")

		operate_layout = QtWidgets.QGridLayout()
		operate_layout.addWidget(self.play_button, *(0, 0))

		self.stop_button = QtWidgets.QToolButton(self)
		self.stop_button.setIcon(QIcon(":icons/stop.png"))
		self.stop_button.setIconSize(QSize(60, 60))
		self.stop_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.stop_button.setObjectName("stop_button")
		self.stop_button.setStyleSheet("border:none")
		operate_layout.addWidget(self.stop_button, *(0, 1))

		baginfo_layout = QtWidgets.QFormLayout()

		test_label = QLabel("调试状态：")
		self.test_status_edit = QLineEdit()
		self.test_status_edit.setReadOnly(True)
		baginfo_layout.addRow(test_label, self.test_status_edit)

		plc_label = QLabel("PLC状态：")
		self.plc_status_edit = QLineEdit()
		self.plc_status_edit.setReadOnly(True)
		baginfo_layout.addRow(plc_label, self.plc_status_edit)
		bagnum_label = QLabel("袋子总数：")
		self.bagnum_edit = QLineEdit()
		baginfo_layout.addRow(bagnum_label, self.bagnum_edit)
		restbaglabel = QLabel("剩余袋子：")
		self.restbagnum_edit = QLineEdit()
		baginfo_layout.addRow(restbaglabel, self.restbagnum_edit)

		currentStatuslabel = QLabel("当前状态：")
		self.currentstatus_edit = QLineEdit()
		baginfo_layout.addRow(currentStatuslabel, self.currentstatus_edit)

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
		self.init_button()  # 按钮状态设置
		self.check_test_status()  # 查验测试状态


		self.plchandle = PlcHandle()

		self.check_plc_status()  # 检验plc状态

		self.init_plc_thread()  # 初始化plc线程

		self.init_imgdetector_thread(IMGHANDLE)  # 初始化图像处理线程

	def init_plc_thread(self):
		'''初始化plc线程'''
		self.plcthread = PlcThread(plchandle=self.plchandle)
		self.plcthread.askforSingnal.connect(self.plc_askposition)
		self.plcthread.moveSignal.connect(self.plc_not_needposition)

	def init_imgdetector_thread(self, IMGHANDLE):
		'''初始化图像处理线程'''
		self.intelligentthread = IntelligentThread(IMGHANDLE=IMGHANDLE, positionservice=PointLocationService(),
		                                           video_player=self.picturelabel)
		self.intelligentthread.positionSignal.connect(self.imgdetector_position_bag_hock)  # 发送移动位置
		self.intelligentthread.dropHockSignal.connect(self.imgdetector_drophock)  # 命令放下钩子
		self.intelligentthread.pullHockSignal.connect(self.imgdetector_pullhock)  # 命令拉起钩子
		self.intelligentthread.findConveyerBeltSignal.connect(self.imgdetector_findconveyerbelt)  # 命令移动袋子到传送带
		self.intelligentthread.dropBagSignal.connect(self.imgdetector_dropbag)
		self.intelligentthread.rebackSignal.connect(self.imgdetector_reback)
		self.intelligentthread.finishSignal.connect(self.imgdetector_finish)
		# self.intelligentthread.rebackSignal.connect(self.afterreback)
		self.intelligentthread.foundbagSignal.connect(self.imgdetector_editbagnum)

	def check_plc_status(self):
		self.plc_status_edit.setText('plc连接成功' if self.plchandle.status else "很抱歉，连接失败")

	def init_button(self):
		self.play_button.clicked.connect(self.play)
		self.stop_button.clicked.connect(self.stop)

	def check_test_status(self):
		self.test_status_edit.setText('测试状态开启' if DEBUG else "测试状态关闭")

	def onClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		print('Key=%s,value=%s' % (item.text(0), item.text(1)))
		filename = os.path.join("D:/video", item.text(0))

		if filename and os.path.isfile(filename) and os.path.exists(filename):
			imagehandle = ImageProvider(videofile=filename, ifsdk= SDK_OPEN)
			self.intelligentthread.IMAGE_HANDLE = imagehandle
			self.play()

	def play(self):
		if self.intelligentthread.IMAGE_HANDLE:
			self.intelligentthread.play = True
			self.intelligentthread.start()
			self.plcthread.work = True
			self.plcthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")

	def autowork(self):
		self.intelligentthread.work = True
		self.plcthread.work = True

	def stop(self):
		'''暂停摄像机'''
		print("关闭摄像")
		self.intelligentthread.play = False

	def plc_askposition(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def plc_not_needposition(self, info: str):
		print(info)
		self.intelligentthread.work = True
		self.plcthread.work = True

	def imgdetector_editbagnum(self, bagnum):
		self.bagnum_edit.setText(str(bagnum))
		print("bag num is {}".format(bagnum))

	def imgdetector_position_bag_hock(self, position):
		'''接收反馈信号'''
		x, y, z = position
		# print("X轴移动：{}，Y轴移动{},z轴移动{}".format(*position))
		# print(position)
		self.plchandle.write_position(position)
		self.plcthread.work = True
		self.intelligentthread.work = True

	def imgdetector_drophock(self, position):
		_x, _y, z = position
		print("已经定位袋子，正在放下钩子,z:{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.DROP_HOCK
		self.currentstatus_edit.setText("已经定位袋子，正在放下钩子")

	def imgdetector_pullhock(self, position):
		_x, _y, z = position
		self.currentstatus_edit.setText("正在拉起袋子,向上拉取{}米".format(z))
		self.plchandle.write_position(position)
		self.intelligentthread.hockstatus = HockStatus.PULL_HOCK

	def imgdetector_findconveyerbelt(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("袋子要到传送带，先向x,y,z各自移动{},{},{}米".format(*position))
		self.intelligentthread.hockstatus = HockStatus.FIND_CONVEYERBELT

	def imgdetector_dropbag(self, position):
		self.plchandle.write_position(position)
		self.currentstatus_edit.setText("移动袋子抵达传送带，向下移动钩子{}米,放下袋子".format(position[2]))
		self.intelligentthread.hockstatus = HockStatus.DROP_BAG

	def imgdetector_reback(self, position):
		self.plchandle.write_position(position)
		print("放下钩子")
		self.currentstatus_edit.setText("复位")
		self.intelligentthread.hockstatus = HockStatus.POSITION_NEARESTBAG

	def imgdetector_finish(self, info):
		# print("工作完成")
		# print(info)
		self.plchandle.ugent_stop()
		self.plcthread.work = False

	def afterreback(self, info):
		'''处理重置'''
		print(info)
		self.plcthread.work = True
		self.intelligentthread.work = False

	def test(self):
		self.check_test_status()


# img = cv2.imread('C:/work/imgs/test/bag6.bmp')
# with PointLocationService(img=img, print_or_no=False) as  a:
# 	a.compute_hook_location()
# # img = a.move()
# img = cv2.resize(img, (800, 800))
# show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
# self.picturelabel.setPixmap(QPixmap.fromImage(showImage))
# self.picturelabel.setScaledContents(True)


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1289, 1000)
		self.centralwidget = CenterWindow()  # 创建一个文本编辑框组件
		self.setCentralWidget(self.centralwidget)  # 将它设置成QMainWindow的中心组件。中心组件占据了所有剩下的空间。
		self.init_menu_toolbar()

	def init_menu_toolbar(self):
		openFileAction = QAction(QIcon(":icons/openfile.png"), '打开', self)
		openFileAction.setShortcut('Ctrl+F')
		openFileAction.setStatusTip('打开文件')
		openFileAction.triggered.connect(self._openfile)

		exitAction = QAction(QIcon(':icons/quit.png'), '退出', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('退出应用')
		exitAction.triggered.connect(self.close)

		openCameraAction = QAction(QIcon(':icons/camera.png'), '摄像头', self)
		openCameraAction.setShortcut('Ctrl+o')
		openCameraAction.setStatusTip('打开摄像头')
		openCameraAction.triggered.connect(self._openCamera)

		stopCameraAction = QAction(QIcon(":icons/close.png"), '关闭摄像头', self)
		stopCameraAction.setShortcut('Ctrl+q')
		stopCameraAction.setStatusTip('关闭摄像头')
		stopCameraAction.triggered.connect(self._stopCamera)

		robotAction = QAction(QIcon(':icons/robot.png'), '自动抓取模式', self)
		robotAction.setShortcut('Ctrl+o')
		robotAction.setStatusTip('自动抓取模式')
		robotAction.triggered.connect(self._work_as_robot)

		testAction = QAction(QIcon(":icons/test.png"), '测试模式', self)
		testAction.setShortcut('Ctrl+t')
		testAction.setStatusTip('测试模式')
		testAction.triggered.connect(self._test)

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

	def _openCamera(self):
		# 正常情况读取sdk
		imagehandle = ImageProvider(ifsdk=SDK_OPEN)
		self.centralwidget.intelligentthread.IMAGE_HANDLE = imagehandle
		self.centralwidget.play()
		self.statusBar().showMessage("已经开启摄像头!")

	# self.statusBar().show()

	def _stopCamera(self):
		'''关闭摄像机'''
		del self.centralwidget.intelligentthread.IMAGE_HANDLE
		self.statusBar().showMessage("已经关闭摄像头!")

	# self.statusBar().show()

	def _work_as_robot(self):
		'''开始智能抓取'''
		self.centralwidget.autowork()
		self.statusBar().showMessage("已经开启智能识别!")

	def _openfile(self):
		filename, filetype = QFileDialog.getOpenFileName(self,
		                                                 "选取文件",
		                                                 "./",
		                                                 "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
		if filename and os.path.isfile(filename) and os.path.exists(filename):
			imagehandle = ImageProvider(videofile=filename, ifsdk=SDK_OPEN)
		self.centralwidget.intelligentthread.IMAGE_HANDLE = imagehandle
		self.centralwidget.play()
		print(filename, filetype)

	def _test(self):
		self.centralwidget.test()

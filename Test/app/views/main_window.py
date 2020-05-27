# -*- coding: utf-8 -*-
import itertools
import os
import re

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit, QSplitter, QListView, \
	QListWidgetItem

from app.config import SDK_OPEN, DEBUG, IMG_WIDTH, IMG_HEIGHT, VIDEO_DIR, ROIS_DIR
from app.core.autowork.process import IntelligentProcess
from app.core.plc.plchandle import PlcHandle
from app.core.video.imageprovider import ImageProvider
from app.icons import resource
from app.views.landmark_window import SetCoordinateWidget

from app.views.roiwindow import SetRoiWidget

SET_ROI = False


class CentWindowUi(object):
	movie_pattern = re.compile("Video_(?P<time>\d+).avi")

	def setupUi(self, Form):
		Form.setObjectName("Form")
		Form.resize(1289, 1000)
		all_layout = QtWidgets.QHBoxLayout()

		self.storedbox = QtWidgets.QGroupBox(self)
		self.storedbox.setObjectName("storedbox")
		self.storedbox.setTitle("已存文档")

		layout = QtWidgets.QVBoxLayout()

		# 构建视频树形列表
		self.init_video_tree()
		layout.addWidget(self.tree)
		self.storedbox.setLayout(layout)

		# all_layout.addWidget(self.roi_box)
		all_layout.addWidget(self.storedbox)

		self.videoBox = QtWidgets.QGroupBox(self)
		self.videoBox.setObjectName("videoBox")
		all_layout.addWidget(self.videoBox)

		self.final_picture_label = QLabel(self)
		self.final_picture_label.setObjectName("final_picture_label")
		# self.final_picture_label.size()
		self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)

		# self.bag_picture_label = QLabel(self)
		# self.bag_picture_label.setObjectName("bag_picture_label")
		# self.bag_picture_label.resize(400, 300)

		# self.laster_picture_label = QLabel(self)
		# self.laster_picture_label.setObjectName("laster_picture_label")
		# self.laster_picture_label.resize(400, 300)

		video_layout = QtWidgets.QGridLayout()
		video_layout.addWidget(self.final_picture_label, 0, 0)
		# video_layout.addWidget(self.bag_picture_label, 0, 1)
		# video_layout.addWidget(self.laster_picture_label, 1, 0)
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

		self.info_box = QtWidgets.QGroupBox()
		self.info_box.setTitle("操作状态")
		plc_status_layout = QtWidgets.QFormLayout()
		test_label = QLabel("调试状态：")
		self.test_status_edit = QLineEdit()
		self.test_status_edit.setReadOnly(True)
		plc_status_layout.addRow(test_label, self.test_status_edit)

		plc_label = QLabel("PLC状态：")
		self.plc_status_edit = QLineEdit()
		self.plc_status_edit.setReadOnly(True)
		plc_status_layout.addRow(plc_label, self.plc_status_edit)

		self.info_box.setLayout(plc_status_layout)

		right_layout = QtWidgets.QVBoxLayout()
		# 添加操作信息
		right_layout.addLayout(operate_layout)
		# 添加袋子信息
		right_layout.addWidget(self.info_box)

		self.operatorBox.setLayout(right_layout)

		self.videoBox.setStyleSheet(''' QLabel{color:black}
		                                QLabel#final_picture_label{background-color:lightgreen}
		                                QLabel#final_picture_label{border:2px}
		                                QLabel#final_picture_label{border-radius:10px}
		                                QLabel#final_picture_label{padding:2px 4px}''')

		all_layout.setStretch(0, 2)
		all_layout.setStretch(1, 7)
		all_layout.setStretch(2, 1)
		self.setLayout(all_layout)
		self.retranslateUi(Form)

	def init_video_tree(self):
		'''
		构建视频树形列表
		:return:
		'''
		try:
			if not os.path.exists(VIDEO_DIR):
				os.makedirs(VIDEO_DIR)
		except:
			pass
		videos = []
		for file in os.listdir(VIDEO_DIR):
			matchresult = re.match(self.movie_pattern, file)
			if matchresult:
				videos.append(matchresult.group(0))
		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(['已存文档'])
		self.tree.setColumnCount(1)
		self.tree.setColumnWidth(0, 200)
		groupinfo = itertools.groupby(videos, key=lambda videofile: videofile[0:10])
		for datestr, files in groupinfo:
			video_level_root = QTreeWidgetItem(self.tree)
			video_level_root.setText(0, datestr)
			video_level_root.setIcon(0, QIcon(":icons/catalogue.png"))
			for filepath in files:
				child = QTreeWidgetItem(video_level_root)
				child.setText(0, filepath)
				# child1.setText(1, 'ios')
				child.setIcon(0, QIcon(":icons/video.png"))
				# child1.setCheckState(0, Qt.Checked)
				video_level_root.addChild(child)
		self.tree.clicked.connect(self.onTreeClicked)
		self.tree.expandAll()

	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("MainWindow", "视频识别机械手"))
		self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
		self.play_button.setText(_translate("MainWindow", "开始"))
		self.stop_button.setText(_translate("MainWindow", "停止"))
		self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QWidget, CentWindowUi):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.init_button()  # 按钮状态设置
		self.plchandle = PlcHandle()
		self.process = IntelligentProcess(IMGHANDLE=None, img_play=self.final_picture_label,plchandle=self.plchandle)

		self.check_test_status()
		self.check_plc_status()

	def init_button(self):
		self.play_button.clicked.connect(self.play)
		self.stop_button.clicked.connect(self.stop)

	def check_test_status(self):
		self.test_status_edit.setText('开启' if DEBUG else "关闭")

	def check_plc_status(self):
		'''检测plc状态'''
		print(self.plchandle.is_open())
		self.plc_status_edit.setText('开启' if self.plchandle.is_open() else "关闭")

	def onTreeClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		filename = os.path.join(VIDEO_DIR, item.text(0))

		if filename and os.path.isfile(filename) and os.path.exists(filename):
			imagehandle = ImageProvider(videofile=filename, ifsdk=False)
			self.process.IMGHANDLE = imagehandle
			self.play()

	def play(self):
		'''开始播放'''
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		if self.process.IMGHANDLE:
			self.process.intelligentthread.play = True
			self.process.intelligentthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")

	def startwork(self):
		'''这是正儿八经的开始移动行车
		'''
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		imagehandle = ImageProvider(videofile=None, ifsdk=True)
		self.process.IMGHANDLE = imagehandle
		if self.process.IMGHANDLE:
			self.process.intelligentthread.play = True
			self.process.intelligentthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")

	def quickly_stop_work(self):
		print("stop work")
		self.process.quickly_stop_work()

	def reset_plc(self):
		'''
		PLC复位
		:return:
		'''
		print("stop work")
		self.process.resetplc()

	def stop(self):
		'''暂停摄像机'''
		print("关闭摄像")
		self.process.intelligentthread.play = False


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1289, 1000)
		self.init_window()
		self.set_roi_widget = SetRoiWidget()

		self.coordinate_widget = SetCoordinateWidget()

	def init_window(self):
		self.setWindowIcon(QIcon(":icons/robot.png"))
		self.centralwidget = CenterWindow()  # 创建一个文本编辑框组件
		self.setCentralWidget(self.centralwidget)  # 将它设置成QMainWindow的中心组件。中心组件占据了所有剩下的空间。
		self.showMaximized()
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

		roisetAction = QAction(QIcon(":icons/set_roi.png"), '选取ROI', self)
		roisetAction.setShortcut('Ctrl+t')
		roisetAction.setStatusTip('选取ROI')
		roisetAction.triggered.connect(self.set_roi)

		setCooridnateAction = QAction(QIcon(":icons/instruct.png"), '设置地标', self)
		setCooridnateAction.setShortcut('Ctrl+t')
		setCooridnateAction.setStatusTip('设置地标')
		setCooridnateAction.triggered.connect(self.set_cooridnate)

		startworkAction = QAction(QIcon(":icons/pointer.png"), '开始工作', self)
		startworkAction.setShortcut('Ctrl+w')
		startworkAction.setStatusTip('开始工作')
		startworkAction.triggered.connect(self.start_work)

		quickly_stop_workAction = QAction(QIcon(":icons/quickly_stop.png"), '紧急停止', self)
		quickly_stop_workAction.setShortcut('Ctrl+s')
		quickly_stop_workAction.setStatusTip('紧急停止')
		quickly_stop_workAction.triggered.connect(self.stop_work)

		reset_plcAction = QAction(QIcon(":icons/recover.png"), '行车复位', self)
		reset_plcAction.setShortcut('Ctrl+r')
		reset_plcAction.setStatusTip('行车复位')
		reset_plcAction.triggered.connect(self.resetplc)

		# recover_workAction = QAction(QIcon(":icons/quickly_stop.png"), '紧急停止', self)
		# quickly_stop_workAction.setShortcut('Ctrl+s')
		# quickly_stop_workAction.setStatusTip('紧急停止')
		# quickly_stop_workAction.triggered.connect(self.stop_work)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&文件')
		fileMenu.addAction(openFileAction)

		roiMenu = menubar.addMenu('&设置ROI')
		roiMenu.addAction(roisetAction)
		cooridnateMenu = menubar.addMenu("&设置坐标系")
		cooridnateMenu.addAction(setCooridnateAction)

		openFileToolBar = self.addToolBar('OpenFile')
		openFileToolBar.addAction(openFileAction)

		exitToolbar = self.addToolBar('Exit')
		exitToolbar.addAction(exitAction)

		setRoiToolbar = self.addToolBar("SetRoi")
		setRoiToolbar.addAction(roisetAction)

		setCooridnateToolBar = self.addToolBar("SetCooridnate")
		setCooridnateToolBar.addAction(setCooridnateAction)

		startWorkToolBar = self.addToolBar("StartWork")
		startWorkToolBar.addAction(startworkAction)

		quicklystopWorkToolBar = self.addToolBar("StopWork")
		quicklystopWorkToolBar.addAction(quickly_stop_workAction)

		resetToolBar = self.addToolBar("ResetPlc")
		resetToolBar.addAction(reset_plcAction)

		self.setWindowTitle('Main window')
		self.statusBar().show()

	def _openfile(self):
		filename, filetype = QFileDialog.getOpenFileName(self,
		                                                 "选取文件",
		                                                 "./",
		                                                 "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
		if filename and os.path.isfile(filename) and os.path.exists(filename):
			if filename.endswith("avi") or filename.endswith("mp4"):
				imagehandle = ImageProvider(videofile=filename, ifsdk=SDK_OPEN)
				self.centralwidget.process.IMGHANDLE = imagehandle
				self.centralwidget.play()
			else:
				self._test()

	def set_roi(self):
		self.set_roi_widget.move(260, 120)
		self.set_roi_widget.show()

	def set_cooridnate(self):
		self.coordinate_widget.move(260, 120)
		self.coordinate_widget.show()

	def start_work(self):
		print("start_work")
		try:
			self.centralwidget.startwork()
		except Exception as e:
			raise e

	def stop_work(self):
		print("quicklystop")
		try:
			self.centralwidget.quickly_stop_work()
		except Exception as e:
			raise e

	def resetplc(self):
		print("resetplc")
		try:
			self.centralwidget.reset_plc()
		except Exception as e:
			raise e

# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import re

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit, QSplitter, QListView, \
	QListWidgetItem, QPushButton, QToolButton

from app.config import SDK_OPEN, DEBUG, IMG_WIDTH, IMG_HEIGHT, VIDEO_DIR, ROIS_DIR, SAVE_VIDEO_DIR, PROGRAM_DATA_DIR
from app.core.autowork.process import IntelligentProcess
from app.core.plc.plchandle import PlcHandle
from app.core.video.imageprovider import ImageProvider
from app.icons import resource
from app.views.commonset_window import CommonSetWidget
from app.views.landmark_window import SetCoordinateWidget

from app.views.roiwindow import SetRoiWidget

SET_ROI = False


class CentWindowUi(object):
	movie_pattern = re.compile("[A-Za-z]+_(?P<time>\d+).avi")

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

		self.up_hock_button = QtWidgets.QToolButton(self)
		self.up_hock_button.setIcon(QIcon(":icons/up.png"))
		self.up_hock_button.setIconSize(QSize(60, 60))
		self.up_hock_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.up_hock_button.setObjectName("up_hock_button")
		self.up_hock_button.setStyleSheet("border:none")
		operate_layout.addWidget(self.up_hock_button, *(1, 0))

		self.down_hock_button = QtWidgets.QToolButton(self)
		self.down_hock_button.setIcon(QIcon(":icons/down.png"))
		self.down_hock_button.setIconSize(QSize(60, 60))
		self.down_hock_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.down_hock_button.setObjectName("down_hock_button")
		self.down_hock_button.setStyleSheet("border:none")
		operate_layout.addWidget(self.down_hock_button, *(1, 1))

		# self.move_east_button = QtWidgets.QToolButton(self)
		# self.move_east_button.setText("东")
		# self.move_east_button.setIcon(QIcon(":icons/down.png"))
		# self.move_east_button.setIconSize(QSize(60, 60))
		# self.move_east_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		# self.move_east_button.setObjectName("move_east_button")
		# # self.move_east_button.setStyleSheet("border:none")
		# operate_layout.addWidget(self.move_east_button, *(2, 0))

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

		ladder_label = QLabel("梯形图状态：")
		self.ladder_edit = QLineEdit()
		self.ladder_edit.setReadOnly(True)
		plc_status_layout.addRow(ladder_label, self.ladder_edit)

		self.fresh_pushbutton = QToolButton()
		self.fresh_pushbutton.setIcon(QIcon(":icons/fresh.png"))
		self.fresh_pushbutton.setIconSize(QSize(60, 60))
		self.fresh_pushbutton.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.fresh_pushbutton.setText("刷新")
		self.fresh_pushbutton.setStyleSheet("border:none")
		self.fresh_pushbutton.clicked.connect(self.fresh_all)
		plc_status_layout.addRow(self.fresh_pushbutton)

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

	def add_save_video(self, save_video_name=None):
		if save_video_name is not None:
			child = QTreeWidgetItem(self.saved_root)
			child.setText(0, save_video_name)
			child.setIcon(0, QIcon(":icons/video.png"))
			self.saved_root.addChild(child)

	def init_video_tree(self):
		'''
		构建视频树形列表
		:return:
		'''
		try:
			if not os.path.exists(VIDEO_DIR):
				os.makedirs(VIDEO_DIR)
			if not os.path.exists(SAVE_VIDEO_DIR):
				os.mkdir(SAVE_VIDEO_DIR)
		except:
			pass
		# haikang_videos = []
		# for file in os.listdir(VIDEO_DIR):
		# 	matchresult = re.match(self.movie_pattern, file)
		# 	# print(matchresult.group(0))
		# 	if matchresult:
		# 		haikang_videos.append(matchresult.group(0))

		save_videos = []
		for item_file in os.listdir(SAVE_VIDEO_DIR):
			matchresult = re.match(self.movie_pattern, item_file)
			# print(matchresult.group(0))
			if matchresult:
				save_videos.append(matchresult.group(0))

		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(['已存文档'])
		self.tree.setColumnCount(1)
		self.tree.setColumnWidth(0, 200)
		# haikang_groupinfo = itertools.groupby(haikang_videos, key=lambda videofile: videofile[0:14])
		# for datestr, files in haikang_groupinfo:
		# 	video_level_root = QTreeWidgetItem(self.tree)
		# 	video_level_root.setText(0, datestr)
		# 	video_level_root.setIcon(0, QIcon(":icons/catalogue.png"))
		# 	for filepath in files:
		# 		child = QTreeWidgetItem(video_level_root)
		# 		child.setText(0, filepath)
		# 		# child1.setText(1, 'ios')
		# 		child.setIcon(0, QIcon(":icons/video.png"))
		# 		# child1.setCheckState(0, Qt.Checked)
		# 		video_level_root.addChild(child)

		self.saved_root = QTreeWidgetItem(self.tree)
		self.saved_root.setText(0, "视频留痕")
		self.saved_root.setIcon(0, QIcon(":icons/catalogue.png"))
		for save_video in save_videos:
			child = QTreeWidgetItem(self.saved_root)
			child.setText(0, save_video)
			child.setIcon(0, QIcon(":icons/video.png"))
			self.saved_root.addChild(child)

		self.tree.clicked.connect(self.onTreeClicked)
		self.tree.expandAll()

	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("MainWindow", "视频识别机械手"))
		self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
		self.play_button.setText(_translate("MainWindow", "开始"))
		self.stop_button.setText(_translate("MainWindow", "停止"))
		self.up_hock_button.setText("上")
		self.down_hock_button.setText("下")
		self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QWidget, CentWindowUi):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.init_button()  # 按钮状态设置
		with open(os.path.join(PROGRAM_DATA_DIR, 'plccom.txt'), 'rb') as comfile:
			info = pickle.load(comfile)
		# print(info)
		self.plchandle = PlcHandle(plc_port=info['PLC_COM'])
		self.process = IntelligentProcess(IMGHANDLE=None, img_play=self.final_picture_label, plchandle=self.plchandle)
		self.process.intelligentthread.update_savevideo.connect(self.add_save_video)
		self.check_test_status()
		self.check_plc_status()
		self.check_ladder_status()

	def init_button(self):
		self.play_button.clicked.connect(self.play)
		self.stop_button.clicked.connect(self.stop)
		self.up_hock_button.clicked.connect(self.up_hock)
		self.down_hock_button.clicked.connect(self.down_hock)

	def check_test_status(self):
		self.test_status_edit.setText('测试' if DEBUG else "正式")

	def check_plc_status(self):
		'''
		检测plc状态
		'''
		print(self.plchandle.is_open())
		self.plc_status_edit.setText('连接' if self.plchandle.is_open() else "断开")

	def check_ladder_status(self):
		'''
		检测梯形图状态
		:return:
		'''
		self.ladder_edit.setText('启动' if self.plchandle.power else "未开启")

	def onTreeClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		filename = os.path.join(SAVE_VIDEO_DIR, item.text(0))

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

	def up_hock(self):
		'''升钩'''
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		if self.process.plchandle:
			print("升钩20公分")
			self.process.plchandle.move(up=20)
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("没有连接PLC!"))

	def down_hock(self):
		'''降钩'''
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		if self.process.plchandle:
			self.process.plchandle.move(down=20)
			print("落钩20公分")
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("没有连接PLC!"))

	def startwork(self):
		'''这是正儿八经的开始移动行车
		'''
		self.process.intelligentthread.work = True
		try:
			imagehandle = ImageProvider(ifsdk=True)
			self.process.IMGHANDLE = imagehandle
			if self.process.IMGHANDLE:
				self.process.IMGHANDLE = imagehandle
				self.process.intelligentthread.play = True
				self.process.intelligentthread.start()
		except:
			QMessageBox.warning(self, "警告",
			                    self.tr("您只是在模拟行车软件，因为没有连接行车摄像头!"))

	# def startwork(self):
	# 	'''这是正儿八经的开始移动行车
	# 	'''
	# 	self.process.intelligentthread.work = True
	# 	try:
	# 		imagehandle = ImageProvider(ifsdk=True)
	# 		self.process.IMGHANDLE = imagehandle
	# 		if self.process.IMGHANDLE:
	# 			self.process.IMGHANDLE = imagehandle
	# 			self.process.intelligentthread.play = True
	# 			self.process.intelligentthread.start()
	# 	except:
	# 		QMessageBox.warning(self, "警告",
	# 		                    self.tr("您只是在模拟行车软件，因为没有连接行车摄像头!"))

	def switch_power(self):
		'''
		启动行车梯形图电源
		:return:
		'''
		print("切换梯形图power")
		self.process.switch_power()

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

	def save_video(self):
		'''
		留存，数据处理留存
		:return:
		'''
		self.process.save_video()

	def stop(self):
		'''暂停摄像机'''
		print("关闭摄像")
		self.process.intelligentthread.play = False

	def fresh_all(self):
		self.check_test_status()
		self.check_plc_status()
		self.check_ladder_status()


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1289, 1000)
		self.init_window()
		self.set_roi_widget = SetRoiWidget()
		self.common_set_widget = CommonSetWidget()
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

		commonAction = QAction(QIcon(":icons/set.png"), '常规设置', self)
		commonAction.setShortcut('Ctrl+i')
		commonAction.setStatusTip('常规设置')
		commonAction.triggered.connect(self.common_set)

		powerAction = QAction(QIcon(":icons/power.png"), '电源', self)
		powerAction.setShortcut('Ctrl+p')
		powerAction.setStatusTip('梯形图启动')
		powerAction.triggered.connect(self.switch_power)

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

		video_save_action = QAction(QIcon(":icons/savevideo.png"), '视频留存', self)
		video_save_action.setShortcut('Ctrl+l')
		video_save_action.setStatusTip('视频留存')
		video_save_action.triggered.connect(self.save_video)

		menubar = self.menuBar()

		fileMenu = menubar.addMenu('&文件')
		fileMenu.addAction(openFileAction)

		basic_site_menu = menubar.addMenu("&基础设置")
		basic_site_menu.addAction(commonAction)  # 常规设置
		basic_site_menu.addAction(setCooridnateAction)  # 定位
		basic_site_menu.addAction(roisetAction)  # roi

		openFileToolBar = self.addToolBar('OpenFile')
		openFileToolBar.addAction(openFileAction)

		exitToolbar = self.addToolBar('Exit')
		exitToolbar.addAction(exitAction)

		powerToolBar = self.addToolBar("StartWork")
		powerToolBar.addAction(powerAction)

		startWorkToolBar = self.addToolBar("StartWork")
		startWorkToolBar.addAction(startworkAction)

		quicklystopWorkToolBar = self.addToolBar("StopWork")
		quicklystopWorkToolBar.addAction(quickly_stop_workAction)

		resetToolBar = self.addToolBar("ResetPlc")
		resetToolBar.addAction(reset_plcAction)

		savevideoToolBar = self.addToolBar("SaveVideo")
		savevideoToolBar.addAction(video_save_action)

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

	def switch_power(self):
		'''
		启动梯形图
		:return:
		'''
		print("切换梯形图电源状态")
		try:
			self.centralwidget.switch_power()
		except Exception as e:
			raise e

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

	def save_video(self):
		try:
			self.centralwidget.save_video()
		except Exception as e:
			raise e

	def common_set(self):
		self.common_set_widget.move(260, 120)
		self.common_set_widget.show()

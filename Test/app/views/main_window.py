# -*- coding: utf-8 -*-
import itertools
import os
import re

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit, QSplitter

from app.config import SDK_OPEN, DEBUG
from app.core.autowork.process import IntelligentProcess
from app.core.video.imageprovider import ImageProvider
from app.icons import resource

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
		try:
			if not os.path.exists("c:/video"):
				os.makedirs("c:/video")
		except:
			pass
		videos = []
		for file in os.listdir("c:/video"):
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
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.init_button()  # 按钮状态设置
		self.process = IntelligentProcess(IMGHANDLE=None, img_play=self.picturelabel,
		                                  plc_status_show=self.plc_status_edit, bag_num_show=self.bagnum_edit)
		self.check_test_status()

	def init_button(self):
		self.play_button.clicked.connect(self.play)
		self.stop_button.clicked.connect(self.stop)

	def check_test_status(self):
		self.test_status_edit.setText('测试状态开启' if DEBUG else "测试状态关闭")

	def onClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		filename = os.path.join("D:/video", item.text(0))

		if filename and os.path.isfile(filename) and os.path.exists(filename):
			imagehandle = ImageProvider(videofile=filename, ifsdk=SDK_OPEN)
			self.process.IMGHANDLE = imagehandle
			self.play()

	def play(self):
		if self.process.IMGHANDLE:
			self.process.intelligentthread.play = True
			self.process.intelligentthread.start()
			self.process.plcthread.work = False
			self.process.plcthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")

	def autowork(self):
		self.process.intelligentthread.work = True
		self.process.plcthread.work = True

	def stop(self):
		'''暂停摄像机'''
		print("关闭摄像")
		self.process.intelligentthread.play = False


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1289, 1000)
		self.setWindowIcon(QIcon(":icons/robot.png"))
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
		self.centralwidget.process.IMGHANDLE = imagehandle
		self.centralwidget.process.intelligentthread.IMAGE_HANDLE = imagehandle
		self.centralwidget.play()
		self.statusBar().showMessage("已经开启摄像头!")

	def _stopCamera(self):
		'''关闭摄像机'''
		del self.centralwidget.process.intelligentthread.IMAGE_HANDLE
		self.statusBar().showMessage("已经关闭摄像头!")

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
			if filename.endswith("avi") or filename.endswith("mp4"):
				imagehandle = ImageProvider(videofile=filename, ifsdk=SDK_OPEN)
				self.centralwidget.process.IMGHANDLE = imagehandle
				self.centralwidget.play()
			else:
				self.centralwidget.process.test(image=filename)

	def _test(self):
		self.centralwidget.process.test()

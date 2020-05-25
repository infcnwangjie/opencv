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
		self.storedbox.setTitle("已存录像")

		layout = QtWidgets.QVBoxLayout()

		# days = ['2020-03-26', '2020-03-27']
		try:
			if not os.path.exists(VIDEO_DIR):
				os.makedirs(VIDEO_DIR)
		except:
			pass
		videos = []
		for file in os.listdir(VIDEO_DIR):
			# if os.path.isfile(file):
			matchresult = re.match(self.movie_pattern, file)
			# print(file)
			if matchresult:
				videos.append(matchresult.group(0))

		self.tree = QTreeWidget()
		self.tree.setHeaderLabels(['视频录像'])
		self.tree.setColumnCount(1)
		self.tree.setColumnWidth(0, 180)
		groupinfo = itertools.groupby(videos, key=lambda videofile: videofile[0:10])
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

		self.tree.clicked.connect(self.onTreeClicked)
		self.tree.expandAll()
		layout.addWidget(self.tree)
		self.storedbox.setLayout(layout)

		all_layout.addWidget(self.storedbox)

		self.videoBox = QtWidgets.QGroupBox(self)
		self.videoBox.setObjectName("videoBox")
		all_layout.addWidget(self.videoBox)

		self.picturelabel = QLabel(self)
		self.picturelabel.setObjectName("picturelabel")
		self.picturelabel.resize(IMG_WIDTH, IMG_HEIGHT)
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

		self.info_box = QtWidgets.QGroupBox()
		self.info_box.setTitle("操作状态")
		baginfo_layout = QtWidgets.QFormLayout()
		test_label = QLabel("调试状态：")
		self.test_status_edit = QLineEdit()
		self.test_status_edit.setReadOnly(True)
		baginfo_layout.addRow(test_label, self.test_status_edit)

		plc_label = QLabel("PLC状态：")
		self.plc_status_edit = QLineEdit()
		self.plc_status_edit.setReadOnly(True)
		baginfo_layout.addRow(plc_label, self.plc_status_edit)
		# bagnum_label = QLabel("袋子总数：")
		# self.bagnum_edit = QLineEdit()
		# baginfo_layout.addRow(bagnum_label, self.bagnum_edit)
		# restbaglabel = QLabel("剩余袋子：")
		# self.restbagnum_edit = QLineEdit()
		# baginfo_layout.addRow(restbaglabel, self.restbagnum_edit)

		currentStatuslabel = QLabel("当前状态：")
		self.currentstatus_edit = QLineEdit()
		baginfo_layout.addRow(currentStatuslabel, self.currentstatus_edit)

		self.info_box.setLayout(baginfo_layout)

		self.roi_box = QtWidgets.QGroupBox(self)
		self.roi_box.setTitle("地标ROI模型")

		self.roi_layout = QtWidgets.QVBoxLayout()

		self.roi_img_listview = QListView()
		self.roilistmodel = QStandardItemModel()
		self.init_roi_imgs()
		self.roi_img_listview.setModel(self.roilistmodel)
		self.roi_layout.addWidget(self.roi_img_listview)
		self.roi_box.setLayout(self.roi_layout)

		right_layout = QtWidgets.QVBoxLayout()
		# 添加操作信息
		right_layout.addLayout(operate_layout)
		# 添加袋子信息
		right_layout.addWidget(self.info_box)

		# slm = QStringListModel()
		right_layout.addWidget(self.roi_box)

		self.operatorBox.setLayout(right_layout)

		self.videoBox.setStyleSheet('''QLabel{color:black}
		                                QLabel{background-color:lightgreen}
		                                QLabel{border:2px}
		                                QLabel{border-radius:10px}
		                             QLabel{padding:2px 4px}''')

		all_layout.setStretch(0, 1.5)
		all_layout.setStretch(1, 6.8)
		all_layout.setStretch(2, 1.7)
		self.setLayout(all_layout)
		self.retranslateUi(Form)

	def init_roi_imgs(self):
		self.roilistmodel.clear()
		for img in os.listdir(ROIS_DIR):
			imgpath = os.path.join(ROIS_DIR, img)
			roi = QStandardItem(QIcon(imgpath), img)
			self.roilistmodel.appendRow(roi)

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
		                                  plc_status_show=self.plc_status_edit)
		self.check_test_status()

	def init_button(self):
		self.play_button.clicked.connect(self.play)
		self.stop_button.clicked.connect(self.stop)

	def check_test_status(self):
		self.test_status_edit.setText('测试状态开启' if DEBUG else "测试状态关闭")


	def onTreeClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		filename = os.path.join(VIDEO_DIR, item.text(0))

		if filename and os.path.isfile(filename) and os.path.exists(filename):
			imagehandle = ImageProvider(videofile=filename, ifsdk=SDK_OPEN)
			self.process.IMGHANDLE = imagehandle
			self.play()

	def play(self):
		'''开始播放'''
		self.picturelabel.resize(IMG_WIDTH, IMG_HEIGHT)
		if self.process.IMGHANDLE:
			self.process.intelligentthread.play = True
			self.process.intelligentthread.start()
			# self.process.plcthread.work = True
			# self.process.plcthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")


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
		self.set_roi_widget.update_listmodel_signal.connect(self.centralwidget.init_roi_imgs)

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

	# def _test(self):
	# 	self.centralwidget.process.monitor_all_once()

	def set_roi(self):

		# global SET_ROI
		# SET_ROI = not SET_ROI
		# img = cv2.imread('d:/2020-04-10-15-26-22test.bmp')
		# # self.centralwidget.process.set_roi(img)
		#
		# dest = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
		# self.centralwidget.picturelabel.set_img(dest)
		# show = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)
		#
		# showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
		# self.centralwidget.picturelabel.setPixmap(QPixmap.fromImage(showImage))
		# self.centralwidget.picturelabel.setScaledContents(True)
		self.set_roi_widget.move(260, 120)
		# self.set_roi_widget.showMaximized()
		self.set_roi_widget.show()

	def set_cooridnate(self):
		self.coordinate_widget.move(260, 120)
		# self.coordinate_widget.showMaximized()
		self.coordinate_widget.show()

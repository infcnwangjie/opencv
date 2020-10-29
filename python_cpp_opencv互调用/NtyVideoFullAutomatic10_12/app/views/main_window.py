# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import re

import cv2
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem, QPainter
from PyQt5.QtWidgets import QMainWindow, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget, QFileDialog, QMessageBox, QDesktopWidget, QLabel, QLineEdit, QSplitter, QListView, \
	QListWidgetItem, QPushButton, QToolButton, QSlider, QLCDNumber, QDockWidget, QGroupBox, QGridLayout, QVBoxLayout, \
	QHBoxLayout
from QCandyUi.CandyWindow import colorful

from app.config import *
from app.core.autowork.process import IntelligentProcess
from app.core.plc.plchandle import PlcHandle
from app.core.video.imageprovider import ImageProvider
from app.icons import resource
from app.views.landmark_window import SetCoordinateWidget, QStyle

from app.views.roiwindow import SetRoiWidget
from app.views.check_error_window import CheckErrorWidget
from app.views.scanbagbyhand import ScanBagByHandWindow
from app.views.show_img_window import ShowWidget
import json

SET_ROI = False
from QCandyUi import CandyWindow
from QCandyUi.CandyWindow import createWindow


def mycolorful(theme, title=None, icon=None):
	"""
	彩色主题装饰, 可以装饰所有的QWidget类使其直接拥有彩色主题 (带Titlebar)
	:param theme: 主题名, 与theme.json里面的主题名对应
	:return:
	"""

	def new_func(aClass):
		def on_call(*args, **kargs):
			src_widget = aClass(*args, **kargs)
			dst_widget = createWindow(src_widget, theme, title, icon)
			dst_widget.showMaximized()
			return dst_widget

		return on_call

	return new_func


# @colorful('blueGreen')
class CenterWindow(QWidget):
	position_pattern = re.compile(".*?(\d+).*?(\d+).*")
	laster_status = 0  # 0是关闭 1是打开
	biglaster_status=0 # 0是关闭 1是打开

	movie_pattern = re.compile("[A-Za-z]+_(?P<time>\d+).avi")
	tree_firsttext_pattern = re.compile("\((.*?),(.*?)\)")

	def __init__(self):
		super().__init__()
		self.bag_positions_childs = []
		self.setupUi()
		# self.dock = QDockWidget("move_direct", self)
		# self.dock.resize(300,300)
		# self.dock_img =QLabel(self.dock)
		# self.dock_img.resize(300,300)
		# self.dock.setWidget(self.dock_img)
		self.plchandle = PlcHandle(plc_port=PLC_COM)
		self.step = 20  # 默认步长
		# self.process = IntelligentProcess(IMGHANDLE=None, img_play=self.final_picture_label, plchandle=self.plchandle,
		#                                   error_widget=self.checkerror_widget,dock_img_player=self.dock_img)
		self.process = IntelligentProcess(IMGHANDLE=None, img_play=self.final_picture_label, plchandle=self.plchandle)
		self.process.intelligentthread.add_scan_bag_signal.connect(self.add_scan_bag)
		self.process.intelligentthread.move_to_bag_signal.connect(self.move_to_bag_slot)
		self.process.intelligentthread.error_show_signal.connect(self.show_error_info)
		self.process.intelligentthread.ariver_advice.connect(self.arrive_show)
		self.process.intelligentthread.detectorhandle.send_warn_info.connect(self.show_warn_info)
		self.init_button()
		self.positions = set()
		self.big = True

	def mousePressEvent(self, event):
		if event.buttons() == Qt.LeftButton:
			if event.buttons() == Qt.LeftButton:
				self.big = not self.big
				if not self.isMaximized():
					self.showMaximized()
				else:
					self.showNormal()

	def setupUi(self):
		self.setObjectName("Form")
		all_layout = QHBoxLayout()

		# self.videoBox = QtWidgets.QGroupBox(self)
		# self.videoBox.setObjectName("videoBox")
		# all_layout.addWidget(self.videoBox)

		self.show_all = ShowWidget()

		# self.final_picture_label = QLabel(self)
		# self.final_picture_label.setObjectName("final_picture_label")
		self.final_picture_label = self.show_all.final_picture_label

		# video_layout = QtWidgets.QGridLayout()
		# video_layout.addWidget(self.show_all, 0, 0)
		# self.videoBox.setLayout(video_layout)

		all_layout.addWidget(self.show_all)

		# 右侧按钮操作区域
		self.operatorBox = QGroupBox(self)
		self.operatorBox.setObjectName("operatorBox")
		all_layout.addWidget(self.operatorBox)

		operator_layout = QVBoxLayout()

		# 准备区
		self.prepareBox = QGroupBox(self)
		self.prepareBox.setObjectName("prepareBox")
		self.prepareBox.setTitle("准备区")
		prepare_layout = QGridLayout()
		self.prepareBox.setLayout(prepare_layout)

		self.load_button = QPushButton(self)
		self.load_button.setToolTip("行车初始化")
		self.load_button.setText("行车加载")
		self.load_button.setIcon(QIcon(":icons/load.png"))
		self.load_button.setIconSize(QSize(40, 40))

		self.load_button.setObjectName("init_button")
		prepare_layout.addWidget(self.load_button, *(0, 0, 1, 1))
		self.open_sdkcamera_button = QPushButton(self)
		self.open_sdkcamera_button.setToolTip("打开摄像")
		self.open_sdkcamera_button.setText("打开摄像")
		self.open_sdkcamera_button.setIcon(QIcon(":icons/camera.png"))
		self.open_sdkcamera_button.setIconSize(QSize(40, 40))
		# self.open_sdkcamera_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.open_sdkcamera_button.setObjectName("open_sdkcamera_button")
		# self.open_sdkcamera_button.setStyleSheet("border:none")
		prepare_layout.addWidget(self.open_sdkcamera_button, 0, 1, 1, 1)
		self.lamp_open_button = QPushButton(self)
		# self.lamp_open_button.setToolTip("打开照明灯")
		self.lamp_open_button.setText("打开照明灯")
		self.lamp_open_button.setIcon(QIcon(":icons/lamp.png"))
		self.lamp_open_button.setIconSize(QSize(40, 40))
		# self.x_center_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.lamp_open_button.setObjectName("lamp_open_button")
		# self.x_center_button.setStyleSheet("border:none")
		prepare_layout.addWidget(self.lamp_open_button, *(0, 2, 1, 1))

		self.turnon_laster_button = QPushButton(self)
		self.turnon_laster_button.setToolTip("开启激光灯")
		self.turnon_laster_button.setText("开启激光灯")
		self.turnon_laster_button.setIcon(QIcon(":icons/laster.png"))
		self.turnon_laster_button.setIconSize(QSize(40, 40))
		self.turnon_laster_button.setObjectName("turnon_laster_button")
		prepare_layout.addWidget(self.turnon_laster_button, *(0, 3, 1, 1))

		self.turnon_biglaster_button = QPushButton(self)
		self.turnon_biglaster_button.setToolTip("开启大射灯")
		self.turnon_biglaster_button.setText("开启大射灯")
		self.turnon_biglaster_button.setIcon(QIcon(":icons/laster.png"))
		self.turnon_biglaster_button.setIconSize(QSize(40, 40))
		self.turnon_biglaster_button.setObjectName("turnon_biglaster_button")
		prepare_layout.addWidget(self.turnon_biglaster_button, *(1, 0, 1, 1))

		self.speed_label = QLabel(self)
		self.speed_label.setText("行车速度")
		prepare_layout.addWidget(self.speed_label, *(2, 0, 1, 1))
		# 步数设置
		self.speed_slide = QSlider(Qt.Horizontal)
		self.speed_slide.setToolTip("设置行车速度")
		# 最小值
		self.speed_slide.setMinimum(650)
		# # 设置最大值
		self.speed_slide.setMaximum(950)
		# self.speed_slide.setSingleStep(10)
		self.speed_slide.setValue(700)
		self.speed_slide.setTickPosition(QSlider.TicksBelow)
		self.speed_slide.setTickInterval(100)
		self.speed_slide.setStyleSheet("""
								QSlider:sub-page:horizontal
								{
								border: 1px solid yellow; 
								background:qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 yellow, stop:1 yellow); 
								border-radius: 3px;
								height: 8px; 
								}
							""")
		prepare_layout.addWidget(self.speed_slide, 2, 1, 1, 2)

		self.speed_value = QLCDNumber(3, self)
		self.speed_value.display(700)

		prepare_layout.addWidget(self.speed_value, 2, 3, 1, 1)

		prepare_layout.setAlignment(Qt.AlignLeft)

		self.prepareBox.setLayout(prepare_layout)
		# self.prepareBox.setStyleSheet("""background: gray;  color: #fff;""")

		# 手动区域
		self.manual_operation_box = QGroupBox()
		self.manual_operation_box.setTitle("手动区")

		manual_operation_layout = QGridLayout()

		self.forward_button = QPushButton(self)
		# self.forward_button.setToolTip("向东")
		self.forward_button.setText("向东")
		# self.forward_button.setIcon(QIcon(":icons/forward.png"))
		self.forward_button.setIconSize(QSize(60, 60))
		# self.forward_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.forward_button.setObjectName("forward_button")
		self.forward_button.setStyleSheet("border:none")

		# manual_operation_layout.addWidget(self.forward_button, *(0, 1))

		manual_operation_layout.addWidget(self.forward_button, *(0, 0))

		self.backward_button = QPushButton(self)
		self.backward_button.setText("向西")
		# self.backward_button.setIcon(QIcon(":icons/backward.png"))
		self.backward_button.setIconSize(QSize(60, 60))
		# self.backward_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.backward_button.setObjectName("backward_button")
		self.backward_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.backward_button, *(0, 1))
		# self.manual_operation_box.resize(300,300)

		self.left_button = QPushButton(self)
		self.left_button.setText("向北")
		# self.left_button.setIcon(QIcon(":icons/left.png"))
		self.left_button.setIconSize(QSize(60, 60))
		# self.left_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.left_button.setObjectName("left_button")
		self.left_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.left_button, *(0, 2))

		self.right_button = QPushButton(self)
		self.right_button.setText("向南")
		# self.right_button.setIcon(QIcon(":icons/right.png"))
		self.right_button.setIconSize(QSize(60, 60))
		# self.right_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.right_button.setObjectName("right_button")
		self.right_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.right_button, *(0, 3))
		self.manual_operation_box.setLayout(manual_operation_layout)
		# self.manual_operation_box.resize(300,300)

		self.up_hock_button = QPushButton(self)
		self.up_hock_button.setText("定位上升")
		# self.up_hock_button.setIcon(QIcon(":icons/up_hock.png"))
		self.up_hock_button.setIconSize(QSize(60, 60))
		# self.up_hock_button.resize(60,60)
		# self.up_hock_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.up_hock_button.setObjectName("up_hock_button")
		self.up_hock_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.up_hock_button, *(1, 0, 1, 1))

		self.down_hock_button = QPushButton(self)
		self.down_hock_button.setText("定位下降")
		# self.down_hock_button.setIcon(QIcon(":icons/down_hock.png"))
		self.down_hock_button.setIconSize(QSize(60, 60))
		# self.down_hock_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.down_hock_button.setObjectName("down_hock_button")
		self.down_hock_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.down_hock_button, 1, 1, 1, 1)
		#

		self.up_cargohook_button = QPushButton(self)
		self.up_cargohook_button.setText("货钩上升")
		# self.up_cargohook_button.setIcon(QIcon(":icons/up_hock.png"))
		self.up_cargohook_button.setIconSize(QSize(60, 60))
		# self.up_cargohook_button.resize(60,60)
		# self.up_cargohook_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.up_cargohook_button.setObjectName("up_cargohook_button")
		self.up_cargohook_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.up_cargohook_button, 1, 2, 1, 1)

		self.down_cargohook_button = QPushButton(self)
		self.down_cargohook_button.setText("货钩下降")
		# self.down_cargohook_button.setIcon(QIcon(":icons/down_hock.png"))
		self.down_cargohook_button.setIconSize(QSize(60, 60))
		# self.down_cargohook_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
		self.down_cargohook_button.setObjectName("down_cargohook_button")
		self.down_cargohook_button.setStyleSheet("border:none")
		manual_operation_layout.addWidget(self.down_cargohook_button, 1, 3, 1, 1)

		self.step_label = QLabel(self)
		self.step_label.setText("设置步长")
		# self.step_label.setStyleSheet("""
		# color: rgb(0,0,255);
		#
		# """)
		manual_operation_layout.addWidget(self.step_label, *(2, 0, 1, 1))
		self.step_slide = QSlider(Qt.Horizontal)
		self.step_slide.setToolTip("设置步数")
		self.step_slide.setMinimum(0)
		self.step_slide.setMaximum(200)
		self.step_slide.setSingleStep(10)
		self.step_slide.setValue(20)
		self.step_slide.setTickPosition(QSlider.TicksBelow)
		self.step_slide.setTickInterval(50)
		self.step_slide.setStyleSheet("""
									QSlider:sub-page:horizontal
									{
									border: 1px solid yellow;
									background:qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 yellow, stop:1 yellow);
									border-radius: 3px;
									height: 8px;
									}
								""")

		manual_operation_layout.addWidget(self.step_slide, *(2, 1, 1, 2))

		self.step_value = QLCDNumber(3, self)
		self.step_value.display(20)

		manual_operation_layout.addWidget(self.step_value, *(2, 3, 1, 1))

		self.manual_operation_box.setLayout(manual_operation_layout)
		# self.manual_operation_box.setStyleSheet("""background: gray;  color: #fff;""")

		# 执行区域
		self.carryout_box = QGroupBox()
		self.carryout_box.setTitle("扫描区")

		carryout_layout = QVBoxLayout(self)

		carryout_horizotal_layout = QGridLayout(self)

		self.zero_button = QPushButton(self)
		self.zero_button.setText("重置")
		self.zero_button.setObjectName("zero_button")
		carryout_horizotal_layout.addWidget(self.zero_button, 0, 0, 1, 1)

		self.clear_button = QPushButton(self)
		self.clear_button.setText("清零")
		self.clear_button.setObjectName("zero_button")
		carryout_horizotal_layout.addWidget(self.clear_button, 0, 1, 1, 1)

		self.stop_button = QPushButton(self)
		self.stop_button.setText("急停")
		self.stop_button.setObjectName("stop_button")
		carryout_horizotal_layout.addWidget(self.stop_button, 0, 2, 1, 1)

		self.scan_button = QPushButton(self)
		self.scan_button.setText("自动扫描袋子")

		self.scan_button.setObjectName("zero_button")

		carryout_horizotal_layout.addWidget(self.scan_button, 1, 0, 1, 1)

		self.scan_handmove_button = QPushButton(self)
		self.scan_handmove_button.setText("手动扫描袋子")

		self.scan_handmove_button.setObjectName("zero_button")
		carryout_horizotal_layout.addWidget(self.scan_handmove_button, 1, 1, 1, 1)

		self.clean_bags_button = QPushButton(self)
		self.clean_bags_button.setText("清空袋子坐标")
		self.clean_bags_button.setObjectName("clean_bags_button")
		carryout_horizotal_layout.addWidget(self.clean_bags_button, 1, 2, 1, 1)

		carryout_layout.addLayout(carryout_horizotal_layout)

		self.carryout_box.setLayout(carryout_layout)

		self.moveandput_box = QGroupBox()
		self.moveandput_box.setTitle("搬运区")
		moveandput_layout = QGridLayout()

		self.build_bagpositions_tree()
		moveandput_layout.addWidget(self.position_tree, 0, 0, 4, 3)

		self.giveup_close_pushbutton = QPushButton(self)
		self.giveup_close_pushbutton.setText("放弃抓取")
		# self.grab_bag_pushbutton.setIcon(QIcon(":icons/grab.png"))
		self.giveup_close_pushbutton.setIconSize(QSize(60, 60))
		self.giveup_close_pushbutton.setObjectName("grab_bag_pushbutton")
		self.giveup_close_pushbutton.setStyleSheet("border:none")
		moveandput_layout.addWidget(self.giveup_close_pushbutton, 4, 0, 1, 1)

		# 抓取袋子
		self.grab_bag_pushbutton = QPushButton(self)
		self.grab_bag_pushbutton.setText("抓取袋子")
		# self.grab_bag_pushbutton.setIcon(QIcon(":icons/grab.png"))
		self.grab_bag_pushbutton.setIconSize(QSize(60, 60))
		self.grab_bag_pushbutton.setObjectName("grab_bag_pushbutton")
		self.grab_bag_pushbutton.setStyleSheet("border:none")
		moveandput_layout.addWidget(self.grab_bag_pushbutton, 4, 1, 1, 1)

		# 放下袋子按钮
		self.putdown_bag_pushbutton = QPushButton(self)
		self.putdown_bag_pushbutton.setText("放下袋子")
		# self.putdown_bag_pushbutton.setIcon(QIcon(":icons/putdown.png"))
		self.putdown_bag_pushbutton.setIconSize(QSize(60, 60))
		self.putdown_bag_pushbutton.setObjectName("putdown_bag_pushbutton")
		self.putdown_bag_pushbutton.setStyleSheet("border:none")
		moveandput_layout.addWidget(self.putdown_bag_pushbutton, 4, 2, 1, 1)

		self.moveandput_box.setLayout(moveandput_layout)

		operator_layout.addWidget(self.prepareBox)
		operator_layout.addWidget(self.manual_operation_box)
		operator_layout.addWidget(self.carryout_box)
		operator_layout.addWidget(self.moveandput_box)
		# operator_layout.addWidget(self.check_error_box)
		self.operatorBox.setLayout(operator_layout)
		# self.operatorBox.setStyleSheet("""background: gray; color: #fff;""")

		self.setStyleSheet("""
									QPushButton
									{
									height: 37px;
									}
								""")

		# all_layout.setStretch(0, 2)
		all_layout.setStretch(0, 7)
		all_layout.setStretch(1, 3)
		self.setLayout(all_layout)
		self.retranslateUi()

	# 构建视频树形列表
	def build_bagpositions_tree(self):

		self.position_tree = QTreeWidget()
		self.position_tree.setHeaderLabels(['袋子坐标', '抵达状态', '抓取状态'])
		self.position_tree.setColumnCount(3)
		self.position_tree.setColumnWidth(0, 150)
		self.position_tree.setColumnWidth(1, 150)
		self.position_tree.setColumnWidth(2, 150)

		self.saved_root = QTreeWidgetItem(self.position_tree)
		self.saved_root.setText(0, "点击执行")
		# self.saved_root.setText(1, "点击执行")
		self.position_tree.clicked.connect(self.onTreeClicked)
		self.position_tree.expandAll()

	def retranslateUi(self):

		self.setWindowTitle("视频识别机械手")
		self.operatorBox.setTitle("操作区域")

	# self.videoBox.setTitle(_translate("MainWindow", "视频区域"))

	def turn_on_off_biglaster(self):
		if self.biglaster_status == 1:
			self.biglaster_status = 0
			self.turnon_biglaster_button.setIcon(QIcon(":icons/turnoff.png"))
			self.turnon_biglaster_button.setText('关闭激光灯')
			self.turnon_biglaster_button.setToolTip("关闭激光灯")
			# self.plchandle.laster = 0
			self.plchandle.biglaster = 0
			self.process.intelligentthread.detectorhandle.laster_status = False
		else:
			self.biglaster_status = 1
			self.plchandle.biglaster = 1
			self.turnon_biglaster_button.setIcon(QIcon(":icons/laster.png"))
			self.turnon_biglaster_button.setText('打开激光灯')
			self.turnon_biglaster_button.setToolTip("打开激光灯")
			self.process.intelligentthread.detectorhandle.laster_status = True

	# 打开或关闭激光灯
	def turn_on_off_laster(self):
		if self.laster_status == 1:
			self.laster_status = 0
			self.turnon_laster_button.setIcon(QIcon(":icons/turnoff.png"))
			self.turnon_laster_button.setText('关闭激光灯')
			self.turnon_laster_button.setToolTip("关闭激光灯")
			self.plchandle.laster = 0
			# self.plchandle.biglaster = 0
			self.process.intelligentthread.detectorhandle.laster_status = False
		else:
			self.laster_status = 1
			self.turnon_laster_button.setIcon(QIcon(":icons/laster.png"))
			self.turnon_laster_button.setText('打开激光灯')
			self.turnon_laster_button.setToolTip("打开激光灯")
			self.plchandle.laster = 1
			self.process.intelligentthread.detectorhandle.laster_status = True

	def arrive_show(self, X, Y):
		info = "->->->已经抵达该袋子,坐标X:{},Y:{}".format(X, Y)
		self.show_all.workrecord.append(info)

		for item in self.bag_positions_childs:
			self.saved_root.removeChild(item)

		self.bag_positions_childs.clear()
		X, Y = int(X), int(Y)

		for index, point in enumerate(self.bagpositions):
			child = QTreeWidgetItem(self.saved_root)
			child.setText(0, "({},{})".format(point[0], point[1]))

			if abs(X - point[0]) < 20 and abs(Y - point[1]) < 20:
				child.setText(1, "已抵达")
			else:
				child.setText(1, "未抵达")
			child.setText(2, "未抓取")
			self.bag_positions_childs.append(child)

	# ------------------------------------------------
	# show_warn_info
	# 功能：展示所有异常
	# 状态：在用
	# 参数： [wraninfo]   ---警告信息
	# 返回： [None]   ---无
	# 作者：王杰  2020-7-*
	# ------------------------------------------------
	def show_warn_info(self, wraninfo):

		infoBox = QMessageBox()
		infoBox.setIcon(QMessageBox.Information)
		infoBox.setText(wraninfo)
		infoBox.setWindowTitle("提示")
		infoBox.setStandardButtons(QMessageBox.Ok)
		infoBox.button(QMessageBox.Ok).animateClick(1 * 1000)  # 3秒自动关闭
		infoBox.exec_()

	def move_to_bag_slot(self, position):
		self.show_all.workrecord.append("\n")
		self.show_all.workrecord.append("命令移到行车到指定位置:({},{})".format(position[0], position[1]))

	# ------------------------------------------------
	# show_error_info
	# 功能：展示所有错误
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-*
	# ------------------------------------------------
	def show_error_info(self, error_info):
		# self.show_all.text.clear()

		if 'south_north' in error_info:
			SOUTH_NORTH_SERVER_FLAG = error_info['south_north']
			# self.show_all.text.append(
			# 	'南北伺服报警' if SOUTH_NORTH_SERVER_FLAG is not None and SOUTH_NORTH_SERVER_FLAG == 1 else "南北伺服正常")
			if SOUTH_NORTH_SERVER_FLAG == 1:
				self.show_all.south_north_server.setIcon(QIcon(":icons/sifu1.png"))

			if 'south' in error_info:
				SOUTH_LIMIT_FLAG = error_info['south']
				# self.show_all.text.append(
				# 	'->->->南限位报警' if SOUTH_LIMIT_FLAG is not None and SOUTH_LIMIT_FLAG == 1 else '->->->南限位正常')
				if SOUTH_LIMIT_FLAG == 1:
					self.show_all.south_xianwei.setIcon(QIcon(":icons/xianweiqi1.png"))

			if 'north' in error_info:
				NORTH_LIMIT_FLAG = error_info['north']
				# self.show_all.text.append(
				# 	'->->->北限位报警' if NORTH_LIMIT_FLAG is not None and NORTH_LIMIT_FLAG == 1 else '->->->北限位正常')
				if NORTH_LIMIT_FLAG == 1:
					self.show_all.north_xianwei.setIcon(QIcon(":icons/xianweiqi1.png"))

		if 'east_server' in error_info:
			EAST_SERVER_FLAG = error_info['east_server']
			# self.show_all.text.append(
			# 	'东伺服报警' if EAST_SERVER_FLAG is not None and EAST_SERVER_FLAG == 1 else '东伺服正常')

			if EAST_SERVER_FLAG == 1:
				self.show_all.east_server.setIcon(QIcon(":icons/sifu1.png"))

			if 'east' in error_info:
				EAST_LIMIT_FLAG = error_info['east']
				# self.show_all.text.append(
				# 	'->->->东限位报警' if EAST_LIMIT_FLAG is not None and EAST_LIMIT_FLAG == 1 else '->->->东限位正常')

				if EAST_LIMIT_FLAG == 1:
					self.show_all.east_xianwei.setIcon(QIcon(":icons/xianweiqi1.png"))

		if 'west_server' in error_info:
			WEST_SERVER_FLAG = error_info['west_server']
			# self.show_all.text.append(
			# 	'西伺服报警' if WEST_SERVER_FLAG is not None and WEST_SERVER_FLAG == 1 else '西伺服正常')
			if WEST_SERVER_FLAG is not None and WEST_SERVER_FLAG == 1:
				self.show_all.west_server.setIcon(QIcon(":icons/sifu1.png"))
			if 'west' in error_info:
				WEST_LIMIT_FLAG = error_info['west']
				# self.show_all.text.append(
				# 	'->->->西限位报警' if WEST_LIMIT_FLAG is not None and WEST_LIMIT_FLAG == 1 else '->->->西限位正常')
				if WEST_LIMIT_FLAG == 1:
					self.show_all.west_xianwei.setIcon(QIcon(":icons/xianweiqi1.png"))

		# 	跳闸报警
		if 'sourth_north_server_trip_warn' in error_info:
			sourth_north_server_trip_warn = error_info['sourth_north_server_trip_warn']
			# self.show_all.text.append(
			# 	'南北伺服跳闸' if sourth_north_server_trip_warn is not None and sourth_north_server_trip_warn == 1 else '南北伺服通电正常')
			if sourth_north_server_trip_warn == 1:
				self.show_all.south_north_server_trip.setIcon(QIcon(":icons/sifu1.png"))

		if 'east_west_server1_trip_warn' in error_info:
			east_west_server1_trip_warn = error_info['east_west_server1_trip_warn']
			# self.show_all.text.append(
			# 	'东西伺服1跳闸' if east_west_server1_trip_warn is not None and east_west_server1_trip_warn == 1 else '东西伺服1通电正常')
			if east_west_server1_trip_warn == 1:
				self.show_all.eastwest_server1_trip.setIcon(QIcon(":icons/dianzha1.png"))

		if 'east_west_server2_trip_warn' in error_info:
			east_west_server2_trip_warn = error_info['east_west_server2_trip_warn']
			# self.show_all.text.append(
			# 	'东西伺服2跳闸' if east_west_server2_trip_warn is not None and east_west_server2_trip_warn == 1 else '东西伺服2通电正常')
			if east_west_server2_trip_warn == 1:
				self.show_all.eastwest_server2_trip.setIcon(QIcon(":icons/dianzha1.png"))

	# ------------------------------------------------
	# 名称：init_button
	# 功能：初始化所有的按钮
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-01
	# ------------------------------------------------
	def init_button(self):
		self.load_button.clicked.connect(self.load_plc_module)
		self.open_sdkcamera_button.clicked.connect(self.open_haikang_camera)
		self.lamp_open_button.clicked.connect(self.open_lamp)
		self.turnon_laster_button.clicked.connect(self.turn_on_off_laster)
		self.turnon_biglaster_button.clicked.connect(self.turn_on_off_biglaster)
		# self.save_video_button.clicked.connect(self.save_video)
		self.speed_slide.valueChanged.connect(self.speed_change)

		self.forward_button.clicked.connect(self.forward)
		self.backward_button.clicked.connect(self.backward)
		self.left_button.clicked.connect(self.left)
		self.right_button.clicked.connect(self.right)

		self.up_hock_button.clicked.connect(self.up_hock)
		self.down_hock_button.clicked.connect(self.down_hock)

		self.up_cargohook_button.clicked.connect(self.up_cargohook)
		self.down_cargohook_button.clicked.connect(self.down_cargohook)

		self.step_slide.valueChanged.connect(self.step_change)

		self.zero_button.clicked.connect(self.reset_plc)
		self.clear_button.clicked.connect(self.clear_plc)
		self.stop_button.clicked.connect(self.quickly_stop_work)
		self.scan_button.clicked.connect(self.scan_bags)
		self.scan_handmove_button.clicked.connect(self.scan_bag_byhand)
		self.clean_bags_button.clicked.connect(self.clean_bag_positions)
		self.giveup_close_pushbutton.clicked.connect(self.giveup_close_bag)
		self.grab_bag_pushbutton.clicked.connect(self.grab_bag)
		self.putdown_bag_pushbutton.clicked.connect(self.putdown_bag)

	# --------------------------------------------------------------------------------
	# 方法名：clean_bag_positions
	# 用途：清除上依次检索的袋子坐标
	# 参数：---None-------
	# 作者：王杰
	# --------------------------------------------------------------------------------#
	def clean_bag_positions(self):
		for item in self.bag_positions_childs:
			self.saved_root.removeChild(item)
		self.process.intelligentthread.detectorhandle.bags.clear()
		self.process.intelligentthread.detectorhandle.bag_detect.bags.clear()
		self.process.intelligentthread.detectorhandle.temp_bag_positions.clear()
		self.process.intelligentthread.detectorhandle.hock_detect.has_stable = False

	def scan_bag_byhand(self):
		self.handmove_window = ScanBagByHandWindow(self.process)
		self.handmove_window.show()

	def add_scan_bag(self, bagpositions: list):
		for item in self.bag_positions_childs:
			self.saved_root.removeChild(item)

		self.show_all.workrecord.clear()

		self.show_all.workrecord.append("扫描到的袋子坐标为：")

		self.bagpositions = bagpositions

		for index, point in enumerate(bagpositions):
			child = QTreeWidgetItem(self.saved_root)
			child.setText(0, "({},{})".format(point[0], point[1]))
			child.setText(1, "未抵达")
			child.setText(2, "未抓取")
			self.bag_positions_childs.append(child)
			self.show_all.workrecord.append("->->->第{}个袋子坐标为：({},{})".format(index + 1, point[0], point[1]))

	def giveup_close_bag(self):
		print("stop")
		self.process.intelligentthread.detectorhandle.keep_y_move = False
		self.process.intelligentthread.detectorhandle.keep_x_move = False
		self.process.intelligentthread.move_close = False
		if self.process.intelligentthread.target_bag_position is not None or len(
				self.process.intelligentthread.target_bag_position) > 0:
			self.process.intelligentthread.detectorhandle.bags.clear()
			self.process.intelligentthread.detectorhandle.bag_detect.bags.clear()
			self.process.intelligentthread.detectorhandle.temp_bag_positions.clear()
			self.process.intelligentthread.detectorhandle.hock_detect.has_stable = False

	def grab_bag(self):
		# TODO  抓取袋子
		if self.process.intelligentthread.move_to_bag_x == False and self.process.intelligentthread.move_to_bag_y == False:
			self.process.intelligentthread.grab_bag = True
			QMessageBox.about(self, '提示', '抓取成功！')
		else:
			QMessageBox.about(self, '错误提示', '还未到达袋子，无法执行放下指令！')

	# TODO 放下袋子
	def putdown_bag(self):

		if self.process.intelligentthread.grab_bag == True and self.process.intelligentthread.putdown_bag == False:
			# TODO  袋子区域计算是实时的，也是十分无序的，想一个完全之策
			# TODO 计算放置袋子的X,Y坐标，以及需要摆放的层数
			# TODO 计算当前袋子与放置坐标的距离
			# TODO 计算最高障碍物高度（不一定准确），让钩子吊着袋子顺利过去
			# TODO 位移到摆放区域X,Y  向下落钩，直到拉力传感器数值为0，控制PLC释放袋子
			self.process.intelligentthread.putdown_bag = True
		else:
			QMessageBox.about(self, '错误提示', '抓取袋子还未成功，无法执行放下指令！')

	# ------------------------------------------------
	# 名称：save_video
	# 功能：录制视频
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-23
	# ------------------------------------------------
	def save_video(self):
		if self.process is not None and self.process.intelligentthread is not None and \
				self.process.intelligentthread.play == True:
			self.process.intelligentthread.save_video = True

	# ------------------------------------------------
	# 名称：open_haikang_camera
	# 功能：打开海康摄像头
	# 状态：在用
	# 参数： [None]
	# 返回： [None]
	# 作者：王杰  2020-7-02
	# ------------------------------------------------
	def open_haikang_camera(self):
		try:
			imagehandle = ImageProvider()
			self.process.IMGHANDLE = imagehandle
			if self.process.IMGHANDLE:
				self.process.IMGHANDLE = imagehandle
				self.process.intelligentthread.play = True
				self.process.intelligentthread.start()
		except Exception as e:
			QMessageBox.about(self, '警告信息', '打开失败')

	# ------------------------------------------------
	# 名称：scan
	# 功能：扫描袋子
	# 状态：在用
	# 参数： [None]
	# 返回： [None]
	# 作者：王杰  2020-7-01
	# ------------------------------------------------
	def scan_bags(self):

		scan_status = self.process.intelligentthread.scan_bag
		self.process.intelligentthread.detectorhandle.bags.clear()
		self.process.intelligentthread.detectorhandle.bag_detect.bags.clear()
		self.process.intelligentthread.detectorhandle.temp_bag_positions.clear()

		if scan_status == False:
			self.process.plchandle.go_and_back()
			self.scan_button.setIcon(QIcon(":icons/stop_scan.png"))
			self.scan_button.setText('停止扫描')
			self.scan_button.setToolTip("停止扫描")
			self.process.intelligentthread.detectorhandle.hock_detect.has_stable = False
		else:
			self.scan_button.setIcon(QIcon(":icons/scan.png"))
			self.scan_button.setText('扫描袋子')
			self.scan_button.setToolTip("扫描袋子")

		self.process.intelligentthread.scan_bag = not scan_status

	def step_change(self):
		value = self.step_slide.value()
		self.step = value
		self.step_value.display(value)

	def speed_change(self):
		value = self.speed_slide.value()
		self.plchandle.speed = value
		self.speed_value.display(value)

	# ------------------------------------------------
	# 名称：forward
	# 功能：向前移动
	# 状态：在用
	# 参数： [None]
	# 返回： [None]
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def forward(self):
		self.process.move(east=self.step)

	# ------------------------------------------------
	# 名称：backward
	# 功能：向后移动
	# 状态：在用
	# 参数： [None]
	# 返回： [None]
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def backward(self):
		self.process.move(west=self.step)

	# ------------------------------------------------
	# 名称：left
	# 功能：向左移动
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def left(self):
		self.process.move(nourth=self.step)

	# ------------------------------------------------
	# 名称：right
	# 功能：向左移动
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def right(self):
		self.process.move(south=self.step)

	# ------------------------------------------------
	# 名称：load_plc_module
	# 功能：开启plc梯形图
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def load_plc_module(self):
		reply = QMessageBox.information(self,  # 使用infomation信息框
		                                "加载",
		                                "加载系统启动参数",
		                                QMessageBox.Yes | QMessageBox.No)
		finish = reload()
		if reply == QMessageBox.Yes:
			self.process.switch_power()
			QMessageBox.about(self, '反馈', '已重新加载系统参数')

	# ------------------------------------------------
	# 名称：open_lamp
	# 功能：打开照明灯
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [None]   ---
	# 作者：王杰  2020-7-1
	# ------------------------------------------------
	def open_lamp(self):
		# reply = QMessageBox.information(self,  # 使用infomation信息框
		#                                 "准备工作",
		#                                 "钩子移动到X轴中心",
		#                                 QMessageBox.Yes | QMessageBox.No)
		# if reply == QMessageBox.Yes:
		# 	self.process.turnon_lamp()
		# QMessageBox.about(self, '反馈', '已置中')
		self.process.turnon_lamp()

	def check_test_status(self):
		self.test_status_edit.setText('测试' if DEBUG else "正式")

	def check_plc_status(self):
		'''
		检测plc状态
		'''
		print(self.plchandle.is_open())
		self.plc_status_edit.setText('连接' if self.plchandle.is_open() else "断开")

	def check_ladder_status(self):
		'''检测梯形图开启状态'''
		self.ladder_edit.setText('启动' if self.plchandle.power else "未开启")

	def onTreeClicked(self, qmodeLindex):
		item = self.position_tree.currentItem()
		self.process.intelligentthread.detectorhandle.hock_detect.has_stable = False
		match_result = re.match(self.position_pattern, item.text(0))
		try:
			if match_result is not None:
				QMessageBox.about(self, '反馈', '定位坐标({},{})'.format(match_result.group(1), match_result.group(2)))
				self.process.intelligentthread.target_bag_position = [match_result.group(1), match_result.group(2)]
				self.process.intelligentthread.move_close = True
				self.process.intelligentthread.move_to_bag_x = True
				self.process.intelligentthread.move_to_bag_y = True

		except:
			print("not match")

	def play(self):
		'''开始播放'''
		if self.process.IMGHANDLE:
			self.process.intelligentthread.play = True
			self.process.intelligentthread.start()
		else:
			QMessageBox.warning(self, "警告",
			                    self.tr("还没有开启摄像头或者选择播放视频!"))
			print("关闭")

	def step_to_millsecond(self):
		# 20cm/1000ms
		speed = 20 / 1000
		millseconds = int(self.step / speed)
		return millseconds

	# 定位钩上升
	def up_hock(self):
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		millseconds = self.step_to_millsecond()
		self.process.move(up=millseconds)

	# 定位钩下降
	def down_hock(self):
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		millseconds = self.step_to_millsecond()
		self.process.move(down=millseconds)

	# 货钩上升
	def up_cargohook(self):
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		millseconds = self.step_to_millsecond()
		self.process.move(up_cargohook=millseconds)

	# 货钩下降
	def down_cargohook(self):
		# self.final_picture_label.resize(IMG_WIDTH, IMG_HEIGHT)
		millseconds = self.step_to_millsecond()
		self.process.move(down_cargohook=millseconds)

	def startwork(self):
		self.process.intelligentthread.work = True
		try:
			if self.process.IMGHANDLE is not None:
				self.process.intelligentthread.play = True
				self.process.intelligentthread.start()
			else:
				imagehandle = ImageProvider(videofile=None)
				self.process.IMGHANDLE = imagehandle
				if self.process.IMGHANDLE:
					self.process.IMGHANDLE = imagehandle
					self.process.intelligentthread.play = True
					self.process.intelligentthread.start()

		except:
			QMessageBox.warning(self, "警告",
			                    self.tr("您只是在模拟行车软件，因为没有连接行车摄像头!"))

	def init_process_status_show(self):
		pass

	# 启动行车梯形图
	def switch_power(self):
		print("切换梯形图power")
		self.process.switch_power()

	def quickly_stop_work(self):
		self.process.quickly_stop_work()

	def re_work(self):
		self.process.re_work()

	# QMessageBox.about(self, '反馈', '紧急停止,防止行车越界')

	def reset_plc(self):
		'''
		PLC复位
		:return:
		'''
		self.process.resetplc()
		QMessageBox.about(self, '反馈', '重置PLC')

	def clear_plc(self):
		'''
		PLC清零
		'''
		self.process.clear_plc()

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
		self.process.intelligentthread.move_to_bag_x = False
		self.process.intelligentthread.move_to_bag_y = False
		self.process.intelligentthread.scan_bag = False
		self.process.intelligentthread.detectorhandle.hock_detect.has_stable = False

	def fresh_all(self):
		self.check_test_status()
		self.check_plc_status()
		self.check_ladder_status()


# blueGreen
@mycolorful('blue', "智能行车", ":icons/robot.png")
class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		# self.resize(1280, 800)
		# self.setWindowTitle("智能行车")

		self.init_window()
		self.set_roi_widget = SetRoiWidget()
		self.coordinate_widget = SetCoordinateWidget()
		self.big = True

	def mousePressEvent(self, event):
		if event.buttons() == Qt.LeftButton:
			if event.buttons() == Qt.LeftButton:
				self.big = not self.big
				if not self.isMaximized():
					self.showMaximized()
				else:
					self.showNormal()

	def init_window(self):
		self.setWindowIcon(QIcon(":icons/robot.png"))

		self.centralwidget = CenterWindow()  # 创建一个文本编辑框组件
		self.setCentralWidget(self.centralwidget)  # 将它设置成QMainWindow的中心组件。中心组件占据了所有剩下的空间。
		# self.showMaximized()
		# self.addDockWidget(Qt.AllDockWidgetAreas, self.centralwidget.dock)
		# self.centralwidget.dock.move(300, 500)
		self.init_menu_toolbar()
		self.setObjectName("MainWindow")

	# self.setStyleSheet("#MainWindow{background-color:rgb(0,255,0)}")

	def close(self):
		if hasattr(self, 'centralwidget'):
			self.centralwidget.clear_plc()

		if self.set_roi_widget.isVisible():
			self.set_roi_widget.close()
		super().close()

	# 退出应用
	def quit_app(self):
		if hasattr(self, 'centralwidget'):
			self.centralwidget.clear_plc()

		if self.set_roi_widget.isVisible():
			self.set_roi_widget.close()
		self.close()

	def init_menu_toolbar(self):
		openFileAction = QAction(QIcon(""), '打开', self)
		openFileAction.setShortcut('Ctrl+F')
		openFileAction.setStatusTip('打开文件')
		openFileAction.triggered.connect(self._openfile)

		exitAction = QAction(QIcon(':icons/quit.png'), '退出', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('退出应用')
		exitAction.triggered.connect(self.quit_app)

		roisetAction = QAction(QIcon(":icons/set_roi.png"), '选取ROI', self)
		roisetAction.setShortcut('Ctrl+t')
		roisetAction.setStatusTip('选取ROI')
		roisetAction.triggered.connect(self.set_roi)

		setCooridnateAction = QAction(QIcon(":icons/instruct.png"), '设置地标', self)
		setCooridnateAction.setShortcut('Ctrl+t')
		setCooridnateAction.setStatusTip('设置地标')
		setCooridnateAction.triggered.connect(self.set_cooridnate)

		# sizechangeAction = QAction(QIcon(":icons/test.png"), '测试模式', self)
		# sizechangeAction.setShortcut('Ctrl+w')
		# sizechangeAction.setStatusTip('测试')
		# sizechangeAction.triggered.connect(self.chang_to_testmode)

		quickly_stop_workAction = QAction(QIcon(":icons/quickly_stop.png"), '紧急停止', self)
		quickly_stop_workAction.setShortcut('Ctrl+s')
		quickly_stop_workAction.setStatusTip('紧急停止')
		quickly_stop_workAction.triggered.connect(self.stop_work)

		reset_plcAction = QAction(QIcon(":icons/recover.png"), '全部重新开始', self)
		reset_plcAction.setShortcut('Ctrl+r')
		reset_plcAction.setStatusTip('行车复位')
		reset_plcAction.triggered.connect(self.re_work)

		menubar = self.menuBar()

		fileMenu = menubar.addMenu('&文件')
		fileMenu.addAction(openFileAction)

		basic_site_menu = menubar.addMenu("&基础设置")
		basic_site_menu.addAction(setCooridnateAction)  # 定位
		basic_site_menu.addAction(roisetAction)  # roi

		openFileToolBar = self.addToolBar('OpenFile')
		openFileToolBar.addAction(openFileAction)

		exitToolbar = self.addToolBar('Exit')
		exitToolbar.addAction(exitAction)

		# testToolBar = self.addToolBar("OpenTestMode")
		# testToolBar.addAction(testAction)

		quicklystopWorkToolBar = self.addToolBar("StopWork")
		quicklystopWorkToolBar.addAction(quickly_stop_workAction)

		resetToolBar = self.addToolBar("ResetPlc")
		resetToolBar.addAction(reset_plcAction)

		# self.setWindowTitle('Main window')
		self.status_bar = self.statusBar()  # 创建状态栏
		self.status_bar.showMessage("ready!")  # 显示消息
		self.centralwidget.process.intelligentthread.detectorhandle.status_show = self.status_bar

	def _openfile(self):
		filename, filetype = QFileDialog.getOpenFileName(self,
		                                                 "选取文件",
		                                                 "./",
		                                                 "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
		if filename and os.path.isfile(filename) and os.path.exists(filename):
			if filename.endswith("avi") or filename.endswith("mp4"):
				imagehandle = ImageProvider(videofile=filename)
				self.centralwidget.process.IMGHANDLE = imagehandle
				self.centralwidget.play()
			else:
				self._test()

	def chang_to_testmode(self):
		print("开启测试模式")

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

	def re_work(self):
		# print("quicklystop")
		try:
			self.centralwidget.re_work()
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

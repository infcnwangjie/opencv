# -*- coding: utf-8 -*-
import os

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize, QRect, QCoreApplication, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QBrush
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QGroupBox, QRadioButton, QHBoxLayout, QWidget, QAction, \
	QTreeWidgetItem, QTreeWidget

from app.config import ICON_DIR
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.core.video.workthread import WorkThread


class CentWindowUi(object):

	def setupUi(self, Form):
		Form.setObjectName("Form")
		Form.resize(1289, 1000)
		all_layout = QtWidgets.QHBoxLayout()

		self.storedbox = QtWidgets.QGroupBox(self)
		self.storedbox.setObjectName("storedbox")
		self.storedbox.setTitle("已存录像")

		layout=QtWidgets.QVBoxLayout()
		self.tree = QTreeWidget()
		# 设置列数
		self.tree.setColumnCount(1)
		# 设置树形控件头部的标题
		self.tree.setHeaderLabels(['录像'])
		# 设置根节点
		root = QTreeWidgetItem(self.tree)
		root.setText(0, '2020-03-26')
		root.setIcon(0, QIcon('./images/root.png'))
		# todo 优化2 设置根节点的背景颜色
		brush_red = QBrush(Qt.red)
		# root.setBackground(0, brush_red)
		brush_blue = QBrush(Qt.blue)
		# root.setBackground(1, brush_blue)
		# 设置树形控件的列的宽度
		self.tree.setColumnWidth(0, 180)
		# 设置子节点1
		child1 = QTreeWidgetItem(root)
		child1.setText(0, '2020-03-26 09:20:31.mp4')
		# child1.setText(1, 'ios')
		child1.setIcon(0, QIcon('./images/IOS.png'))
		# child1.setCheckState(0, Qt.Checked)
		root.addChild(child1)
		# 设置子节点2
		child2 = QTreeWidgetItem(root)
		child2.setText(0, '2020-03-26 11:22:31.mp4')
		# child2.setText(1, '')
		child2.setIcon(0, QIcon('./images/android.png'))
		# 设置子节点3
		child3 = QTreeWidgetItem(root)
		child3.setText(0, '2020-03-26 16:22:31.mp4')
		# child3.setText(1, 'android')
		child3.setIcon(0, QIcon('./images/music.png'))
		# 加载根节点的所有属性与子控件
		self.tree.addTopLevelItem(root)
		self.tree.clicked.connect(self.onClicked)
		# 节点全部展开
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

		self.open_camera_button = QtWidgets.QPushButton(self)
		self.open_camera_button.setMaximumSize(QSize(130, 41))
		self.open_camera_button.setIcon(QIcon(os.path.join(ICON_DIR, "camera.png")))
		self.open_camera_button.setObjectName("open_camera_button")

		operate_layout = QtWidgets.QGridLayout()
		operate_layout.addWidget(self.open_camera_button, *(0, 0))

		self.auto_work_button = QtWidgets.QPushButton(self)
		self.auto_work_button.setMaximumSize(QSize(130, 41))
		self.auto_work_button.setIcon(QIcon(os.path.join(ICON_DIR, "auto.png")))
		self.auto_work_button.setObjectName("auto_work_button")

		operate_layout.addWidget(self.auto_work_button, *(0, 1))
		self.stop_work_button = QtWidgets.QPushButton(self)
		self.stop_work_button.setMaximumSize(QSize(130, 41))
		self.stop_work_button.setIcon(QIcon(os.path.join(ICON_DIR, "stop.png")))
		self.stop_work_button.setObjectName("stop_work_button")
		operate_layout.addWidget(self.stop_work_button, *(1, 0))

		self.stop_camera_button = QtWidgets.QPushButton(self)
		self.stop_camera_button.setMaximumSize(QSize(130, 41))
		self.stop_camera_button.setIcon(QIcon(os.path.join(ICON_DIR, "close.png")))
		operate_layout.addWidget(self.stop_camera_button, *(1, 1))

		self.quit_button = QtWidgets.QPushButton(self)
		self.quit_button.setMaximumSize(QSize(130, 41))
		self.quit_button.setIcon(QIcon(os.path.join(ICON_DIR, "quit.png")))
		self.quit_button.setObjectName("quit_button")
		operate_layout.addWidget(self.quit_button, *(2, 0))

		self.test_button = QtWidgets.QPushButton(self)
		self.test_button.setMaximumSize(QSize(130, 41))
		self.test_button.setIcon(QIcon(os.path.join(ICON_DIR, "test.png")))
		self.test_button.setObjectName("test")
		operate_layout.addWidget(self.test_button, *(2, 1))

		baginfo_layout = QtWidgets.QFormLayout()

		# 添加袋子信息
		right_layout = QtWidgets.QVBoxLayout()
		right_layout.addLayout(baginfo_layout)
		right_layout.addLayout(operate_layout)

		self.operatorBox.setLayout(right_layout)

		self.videoBox.setStyleSheet('''QLabel{color:black}
		                                QLabel{background-color:lightgreen}
		                                QLabel{border:2px}
		                                QLabel{border-radius:10px}
		                             QLabel{padding:2px 4px}''')

		all_layout.setStretch(0,2)
		all_layout.setStretch(1, 6)
		all_layout.setStretch(2, 2)
		self.setLayout(all_layout)
		self.retranslateUi(Form)

	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("MainWindow", "视频识别机械手"))
		self.operatorBox.setTitle(_translate("MainWindow", "操作区域"))
		self.open_camera_button.setText(_translate("MainWindow", "开启摄像"))
		self.stop_work_button.setText(_translate("MainWindow", "停止智能"))
		self.auto_work_button.setText(_translate("MainWindow", "开启智能"))
		self.stop_camera_button.setText(_translate("MainWindow", "关闭摄像"))
		self.quit_button.setText(_translate("MainWindow", "退出系统"))
		self.test_button.setText(_translate("MainWindow", "测试"))
		self.videoBox.setTitle(_translate("MainWindow", "视频区域"))


class CenterWindow(QWidget, CentWindowUi):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.open_camera_button.clicked.connect(self.opencamera)
		self.auto_work_button.clicked.connect(self.autowork)
		self.stop_work_button.clicked.connect(self.stopwork)
		self.stop_camera_button.clicked.connect(self.stop_camera)
		self.quit_button.clicked.connect(QCoreApplication.quit)
		self.test_button.clicked.connect(self.test)
		self.thread = WorkThread(video_file="D:/video/test.mp4", video_player=self.picturelabel)
		self.thread.sinOut.connect(self.info)

	def changeEvent(self, e):
		if e.type() == QtCore.QEvent.WindowStateChange:
			if self.isMinimized():
				print("窗口最小化")
			elif self.isMaximized():
				print("窗口最大化")
			# desktop = QDesktopWidget()
			# screen_width = desktop.screenGeometry().width()
			# screen_height = desktop.screenGeometry().height()
			# self.resize(QSize(screen_width,screen_height))
			# print(screen_width, screen_height)
			# self.picturelabel.resize(QSize(screen_width * 0.7 - 20, screen_height * 0.75))
			# self.operatorBox.resize(QSize(screen_width*0.3, screen_height * 0.8))
			# self.videoBox.resize(QSize(screen_width * 0.7, screen_height * 0.8))

			elif self.isFullScreen():
				print("全屏显示")
			elif self.isActiveWindow():
				print("活动窗口")
		QtWidgets.QWidget.changeEvent(self, e)

	def onClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		print('Key=%s,value=%s' % (item.text(0), item.text(1)))

	def opencamera(self):
		# img = cv2.imread('C:/work/imgs/test/bag6.bmp')
		# with PointLocationService(img=img, print_or_no=False) as  a:
		# 	a.computelocations(flag=BAG_AND_LANDMARK)
		# 	img = a.move()
		# show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
		# self.picturelabel.setPixmap(QPixmap.fromImage(showImage))
		# self.picturelabel.setScaledContents(True)
		self.thread.playing = True
		self.thread.start()

	def autowork(self):
		'''开启智能工作'''
		# self.thread.start()
		print("开启工作")

	def stopwork(self):
		'''关闭智能工作'''
		self.thread.stopwork()

	def stop_camera(self):
		'''暂停摄像机'''
		print("关闭摄像")
		self.thread.stopcamera()

	def info(self, infomessage):
		'''接收反馈信号'''
		print(infomessage)

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


# self.picturelabel.resize(QSize(screen_width * 0.7 - 20, screen_height * 0.75))
# self.operatorBox.resize(QSize(screen_width*0.3, screen_height * 0.8))
# self.videoBox.resize(QSize(screen_width * 0.7, screen_height * 0.8))


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.resize(1289, 1000)
		center = CenterWindow()  # 创建一个文本编辑框组件
		self.setCentralWidget(center)  # 将它设置成QMainWindow的中心组件。中心组件占据了所有剩下的空间。
		self.menu_toolbar_ui()

	def menu_toolbar_ui(self):
		exitAction = QAction(QIcon(os.path.join(ICON_DIR, 'quit.png')), '退出', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setStatusTip('退出应用')
		exitAction.triggered.connect(self.close)

		openCameraAction = QAction(QIcon(os.path.join(ICON_DIR, 'camera.png')), '摄像头', self)
		openCameraAction.setShortcut('Ctrl+o')
		openCameraAction.setStatusTip('打开摄像头')
		openCameraAction.triggered.connect(lambda: print("打开摄像头"))

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&文件')
		fileMenu.addAction(exitAction)

		openCameraMenu = menubar.addMenu('&摄像头')
		openCameraMenu.addAction(openCameraAction)

		toolbar = self.addToolBar('Exit')
		toolbar.addAction(exitAction)

		openCameraToolbar = self.addToolBar("OpenCamera")
		openCameraToolbar.addAction(openCameraAction)

		# self.setGeometry(300, 300, 1200, 900)
		self.setWindowTitle('Main window')
		self.show()

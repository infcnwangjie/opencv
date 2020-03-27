import os

import cv2
from PyQt5.QtCore import Qt, QSize
from PyQt5.uic.properties import QtWidgets

from app.config import ICON_DIR
from PyQt5 import QtCore, QtWidgets

from app.views.main_window import CenterWindow


def video_write():
	img = cv2.imread('D:/imgs/merge/1.bmp')
	imgInfo = img.shape
	size = (imgInfo[1], imgInfo[0])
	print(size)
	videoWrite = cv2.VideoWriter('D:/video/test.mp4', -1, 15, size)  # 写入对象 1 file name
	# 2 编码器 3 帧率 4 size
	for imgfile in os.listdir("D:/imgs/merge"):
		fileName = os.path.join("D:/imgs/merge", imgfile)
		img = cv2.imread(fileName)
		videoWrite.write(img)  # 写入方法 1 jpg data
	print("end")



import sys
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication, QPushButton, QTreeWidgetItem, QTreeWidget,QWidget
from PyQt5.QtGui import QIcon, QBrush


class LeftPartWidget(QWidget):
	def __init__(self):
		super().__init__()
		self.setui()
		self.resize(QSize(200, 1000))

	def setui(self):
		self.treebox = QtWidgets.QGroupBox(self)

		self.treebox.setTitle("已存录像")
		cent_layout = QtWidgets.QHBoxLayout()
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
		self.tree.setColumnWidth(0, 100)
		# 设置子节点1
		child1 = QTreeWidgetItem(root)
		child1.setText(0, '2020-03-26 09:20:31.mp4')
		# child1.setText(1, 'ios')
		child1.setIcon(0, QIcon('./images/IOS.png'))
		# todo 优化1 设置节点的状态
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
		# self.setCentralWidget(self.tree)
		# self.left_layout.addWidget(self.tree)
		# self.setLayout(self.all_layout)

		cent_layout.addWidget(self.tree)

		self.treebox.setLayout(cent_layout)

	def onClicked(self, qmodeLindex):
		item = self.tree.currentItem()
		print('Key=%s,value=%s' % (item.text(0), item.text(1)))



class Example(QMainWindow):
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


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())

import os

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
import app.icons.resource
import pickle

from app.config import PROGRAM_DATA_DIR


class SetCoordinateWidget(QWidget):
	def __init__(self):
		super().__init__(parent=None)
		self.setupUi(self)
		self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint)

	def setupUi(self, CoordirateWidget):
		CoordirateWidget.setObjectName("CoordirateWidget")
		CoordirateWidget.resize(1032, 871)

		self.scroll_area = QScrollArea(self)  # 2

		self.groupBox = QtWidgets.QGroupBox(CoordirateWidget)
		self.groupBox.setGeometry(QtCore.QRect(30, 20, 961, 731))
		self.groupBox.setObjectName("groupBox")
		self.scroll_area.setWidget(self.groupBox)
		self.scroll_area.setMinimumSize(QSize(1000, 680))
		# self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self.left_toolButton1 = QtWidgets.QToolButton(self.groupBox)
		self.left_toolButton1.setGeometry(QtCore.QRect(120, 50, 81, 101))
		self.left_toolButton1.setObjectName("toolButton")
		self.right_toolButton1 = QtWidgets.QToolButton(self.groupBox)
		self.right_toolButton1.setGeometry(QtCore.QRect(670, 60, 81, 101))
		self.right_toolButton1.setObjectName("right_toolButton1")
		self.left_toolButton2 = QtWidgets.QToolButton(self.groupBox)
		self.left_toolButton2.setGeometry(QtCore.QRect(120, 210, 81, 101))
		self.left_toolButton2.setObjectName("right_toolButton2")
		self.right_toolButton2 = QtWidgets.QToolButton(self.groupBox)
		self.right_toolButton2.setGeometry(QtCore.QRect(670, 220, 81, 101))
		self.right_toolButton2.setObjectName("left_toolButton2")

		self.left_toolButton3 = QtWidgets.QToolButton(self.groupBox)
		self.left_toolButton3.setGeometry(QtCore.QRect(120, 380, 81, 101))
		self.left_toolButton3.setObjectName("left_toolButton3")
		self.right_toolButton3 = QtWidgets.QToolButton(self.groupBox)
		self.right_toolButton3.setGeometry(QtCore.QRect(670, 380, 81, 101))
		self.right_toolButton3.setObjectName("right_toolButton3")
		self.left_toolButton4 = QtWidgets.QToolButton(self.groupBox)
		self.left_toolButton4.setGeometry(QtCore.QRect(120, 570, 81, 101))
		self.left_toolButton4.setObjectName("left_toolButton4")
		self.right_toolButton4 = QtWidgets.QToolButton(self.groupBox)
		self.right_toolButton4.setGeometry(QtCore.QRect(670, 570, 81, 101))
		self.right_toolButton4.setObjectName("right_toolButton4")

		self.lineEditL1 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL1.setGeometry(QtCore.QRect(210, 110, 151, 41))
		self.lineEditL1.setObjectName("lineEditL1")
		self.lineEditL1.setText('0,0')

		self.lineEditR1 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR1.setGeometry(QtCore.QRect(760, 120, 151, 41))
		self.lineEditR1.setObjectName("lineEditR1")
		self.lineEditR1.setText('5.743,0')
		self.lineEditL2 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL2.setGeometry(QtCore.QRect(210, 270, 151, 41))
		self.lineEditL2.setObjectName("lineEditL2")
		self.lineEditL2.setText('0,0.7')
		self.lineEditR2 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR2.setGeometry(QtCore.QRect(760, 280, 151, 41))
		self.lineEditR2.setObjectName("lineEditR2")
		self.lineEditR2.setText('5.743,0.7')
		self.lineEditL3 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL3.setGeometry(QtCore.QRect(210, 440, 151, 41))
		self.lineEditL3.setObjectName("lineEditL3")
		self.lineEditL3.setText("0,1.7")
		self.lineEditR3 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR3.setGeometry(QtCore.QRect(760, 450, 151, 41))
		self.lineEditR3.setObjectName("lineEditR3")
		self.lineEditR3.setText("5.743,1.7")
		self.lineEditL4 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL4.setGeometry(QtCore.QRect(210, 630, 151, 41))
		self.lineEditL4.setObjectName("lineEditL4")
		self.lineEditL4.setText("0,3.7")
		self.lineEditR4 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR4.setGeometry(QtCore.QRect(760, 630, 151, 41))
		self.lineEditR4.setObjectName("lineEditR4")
		self.lineEditR4.setText("5.743,3.7")

		self.save_coordirate_button = QtWidgets.QPushButton(CoordirateWidget)
		self.save_coordirate_button.setGeometry(QtCore.QRect(670, 790, 241, 51))
		self.save_coordirate_button.setObjectName("save_coordirate_button")

		# scroll = QScrollArea(CoordirateWidget)
		# scroll.setWidget(self.groupBox)
		# scroll.setWidgetResizable(True)
		# scroll.setFixedHeight(400)
		self.save_coordirate_button.clicked.connect(self.save_click)
		self.retranslateUi(CoordirateWidget)
		QtCore.QMetaObject.connectSlotsByName(CoordirateWidget)

	def retranslateUi(self, CoordirateWidget):
		_translate = QtCore.QCoreApplication.translate
		CoordirateWidget.setWindowTitle(_translate("CoordirateWidget", "行车车间坐标系"))
		self.left_toolButton1.setText(_translate("CoordirateWidget", "NO1_L"))
		self.right_toolButton1.setText(_translate("CoordirateWidget", "NO1_R"))
		self.left_toolButton2.setText(_translate("CoordirateWidget", "NO2_L"))
		self.right_toolButton2.setText(_translate("CoordirateWidget", "NO2_R"))
		self.left_toolButton3.setText(_translate("CoordirateWidget", "NO3_L"))
		self.right_toolButton3.setText(_translate("CoordirateWidget", "NO3_R"))
		self.left_toolButton4.setText(_translate("CoordirateWidget", "NO4_L"))
		self.right_toolButton4.setText(_translate("CoordirateWidget", "NO4_R"))
		self.groupBox.setTitle(_translate("CoordirateWidget", "车间实景坐标"))
		self.save_coordirate_button.setText(_translate("CoordirateWidget", "保存"))

	def save_click(self):
		info = dict(left_1=self.lineEditL1.text(), right_1=self.lineEditR1.text(),
		            left_2=self.lineEditL2.text(), right_2=self.lineEditR2.text(),
		            left_3=self.lineEditL3.text(), right_3=self.lineEditR3.text(),
		            left_4=self.lineEditL4.text(), right_4=self.lineEditR4.text()
		            )
		# print(info)
		with open(os.path.join(PROGRAM_DATA_DIR,'coordinate_data.txt'),'wb') as coordinate:
			pickle.dump(info,coordinate,0)
		# with open(os.path.join(PROGRAM_DATA_DIR,'coordinate_data.txt'),'rb')  as file:
		#
		# 	mydata=pickle.load(file)
		# 	print(mydata)



if __name__ == '__main__':
	import sys

	app = QApplication(sys.argv)
	form = SetCoordinateWidget()
	form.show()
	app.exec_()

import os
import pickle

from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import *
import sys
import app.icons.resource
from app.config import PROGRAM_DATA_DIR


class SetCoordinateWidget(QWidget):
	def __init__(self, row=8, col_num=6):
		super().__init__(parent=None)
		self.row_num = row
		self.col_num = col_num
		# self.rectangle_ruler(self.row_num, self.col_num)
		self.setupUi(self)
		self.setWindowIcon(QIcon(":icons/instruct.png"))

	def rectangle_ruler(self, rows, cols):
		widget_width = 95
		colunm_interval = 5

		widget_height = 95
		row_interval = 5

		for row in range(0, rows):
			for col in range(0, cols):
				widget = QWidget(self)
				col_interval_nm, col_widget_nm = col + 1, col + 1  # 每列间隔数目也要计算清楚,每列方块数目也要计算清楚

				row_interval_nm, row_widget_nm = row + 1, row + 1
				print(col_interval_nm * colunm_interval + col_widget_nm * widget_width,
				      row_interval_nm * row_interval + row_widget_nm * widget_height)

				widget.setGeometry(col_interval_nm * colunm_interval + col_widget_nm * widget_width, \
				                   row_interval_nm * row_interval + row_widget_nm * widget_height, \
				                   widget_width, widget_height)

				widget.setStyleSheet('background-color:#9AFF9A')

	def setupUi(self, CoordirateWidget):

		CoordirateWidget.setObjectName("CoordirateWidget")
		# self.scroll_area = QScrollArea(self)  # 2

		self.groupBox = QtWidgets.QGroupBox(CoordirateWidget)
		self.groupBox.setGeometry(QtCore.QRect(100, 100, 961, 731))
		self.groupBox.setObjectName("groupBox")
		#
		# self.scroll_area.setWidget(self.groupBox)
		# self.scroll_area.setMinimumSize(QSize(1000, 680))
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
		self.lineEditR1.setText('574.3,0')
		self.lineEditL2 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL2.setGeometry(QtCore.QRect(210, 270, 151, 41))
		self.lineEditL2.setObjectName("lineEditL2")
		self.lineEditL2.setText('0,70')
		self.lineEditR2 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR2.setGeometry(QtCore.QRect(760, 280, 151, 41))
		self.lineEditR2.setObjectName("lineEditR2")
		self.lineEditR2.setText('574.3,70')
		self.lineEditL3 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL3.setGeometry(QtCore.QRect(210, 440, 151, 41))
		self.lineEditL3.setObjectName("lineEditL3")
		self.lineEditL3.setText("0,170")
		self.lineEditR3 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR3.setGeometry(QtCore.QRect(760, 450, 151, 41))
		self.lineEditR3.setObjectName("lineEditR3")
		self.lineEditR3.setText("574.3,170")
		self.lineEditL4 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditL4.setGeometry(QtCore.QRect(210, 630, 151, 41))
		self.lineEditL4.setObjectName("lineEditL4")
		self.lineEditL4.setText("0,370")
		self.lineEditR4 = QtWidgets.QLineEdit(self.groupBox)
		self.lineEditR4.setGeometry(QtCore.QRect(760, 630, 151, 41))
		self.lineEditR4.setObjectName("lineEditR4")
		self.lineEditR4.setText("574.3,370")

		self.save_coordirate_button = QtWidgets.QPushButton(CoordirateWidget)
		self.save_coordirate_button.setGeometry(QtCore.QRect(100, 840, 241, 51))
		self.save_coordirate_button.setObjectName("save_coordirate_button")

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
		info = dict(NO1_L=self.lineEditL1.text(), NO1_R=self.lineEditR1.text(),
		            NO2_L=self.lineEditL2.text(), NO2_R=self.lineEditR2.text(),
		            NO3_L=self.lineEditL3.text(), NO3_R=self.lineEditR3.text(),
		            NO4_L=self.lineEditL4.text(), NO4_R=self.lineEditR4.text()
		            )
		# print(info)
		with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'wb') as coordinate:
			pickle.dump(info, coordinate, 0)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	d = SetCoordinateWidget()
	d.show()
	sys.exit(app.exec_())

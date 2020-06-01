# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'commonset.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
import pickle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QApplication

from app.config import PROGRAM_DATA_DIR
from app.icons import resource


class CommonsetUi(object):

	def setupUi(self, Form):
		'''
		设置UI
		:param Form:
		:return:
		'''
		Form.setObjectName("Form")
		Form.resize(300, 200)
		self.topwidget = QtWidgets.QWidget(Form)
		self.topwidget.setGeometry(QtCore.QRect(20, 50, 200, 101))
		self.topwidget.setObjectName("topwidget")
		self.formLayout = QtWidgets.QFormLayout(self.topwidget)
		self.formLayout.setContentsMargins(0, 0, 0, 0)
		self.formLayout.setObjectName("formLayout")
		# self.data_save_label = QtWidgets.QLabel(self.topwidget)
		# self.data_save_label.setObjectName("data_save_label")
		# self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.data_save_label)
		# self.data_save_edit = QtWidgets.QLineEdit(self.topwidget)
		# self.data_save_edit.setObjectName("data_save_edit")
		# self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.data_save_edit)
		self.plc_com_label = QtWidgets.QLabel(self.topwidget)
		self.plc_com_label.setObjectName("plc_com_label")
		self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.plc_com_label)
		self.plc_com_edit = QtWidgets.QLineEdit(self.topwidget)
		self.plc_com_edit.setObjectName("plc_com_edit")
		self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.plc_com_edit)
		self.bottomwidget = QtWidgets.QWidget(Form)
		self.bottomwidget.setGeometry(QtCore.QRect(20, 120, 120, 30))
		self.bottomwidget.setObjectName("bottomwidget")
		self.horizontalLayout = QtWidgets.QHBoxLayout(self.bottomwidget)
		self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.save_pushbutton = QtWidgets.QPushButton(self.bottomwidget)
		self.save_pushbutton.setObjectName("save_pushbutton")
		self.horizontalLayout.addWidget(self.save_pushbutton)
		self.cancel_pushbutton = QtWidgets.QPushButton(self.bottomwidget)
		self.cancel_pushbutton.setObjectName("cancel_pushbutton")
		self.horizontalLayout.addWidget(self.cancel_pushbutton)

		self.retranslateUi(Form)
		QtCore.QMetaObject.connectSlotsByName(Form)

	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("Form", "Form"))
		# self.data_save_label.setText(_translate("Form", "数据存储文件："))
		self.plc_com_label.setText(_translate("Form", "PLC端口:"))
		self.save_pushbutton.setText(_translate("Form", "保存"))
		self.cancel_pushbutton.setText(_translate("Form", "取消"))


class CommonSetWidget(QWidget):
	def __init__(self):
		super().__init__(parent=None)
		self.ui = CommonsetUi()
		self.ui.setupUi(self)
		self.setWindowIcon(QIcon(":icons/set.png"))
		self.ui.save_pushbutton.clicked.connect(self.save)
		self.ui.cancel_pushbutton.clicked.connect(self.cancel)

	def save(self):
		'''
		保存
		:return:
		'''
		print(PROGRAM_DATA_DIR)
		info = dict(PLC_COM=self.ui.plc_com_edit.text())
		with open(os.path.join(PROGRAM_DATA_DIR, 'plccom.txt'), 'wb') as comfile:
			pickle.dump(info, comfile, 0)

	def cancel(self):
		'''
		取消
		:return:
		'''
		self.ui.plc_com_edit.text('')


if __name__ == '__main__':
	# import sys
	#
	# app = QApplication(sys.argv)
	# form = CommonSetWidget()
	# form.show()
	# app.exec_()
	with open(os.path.join(PROGRAM_DATA_DIR, 'plccom.txt'), 'rb') as comfile:
		info=pickle.load(comfile)
		print(info)

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SetUi.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class SetRoiUi(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1228, 768)
        self.roi_label = QtWidgets.QLabel(Form)
        self.roi_label.setGeometry(QtCore.QRect(30, 50, 900, 700))
        self.roi_label.setObjectName("roi_label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(1080, 310, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(1090, 450, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.roi_label.setText(_translate("Form", "TextLabel"))
        self.pushButton.setText(_translate("Form", "开始选择roi"))
        self.pushButton_2.setText(_translate("Form", "保存"))

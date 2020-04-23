# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\git\opencv\SetCoordirateUi.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CoordirateWidget(object):
    def setupUi(self, CoordirateWidget):
        CoordirateWidget.setObjectName("CoordirateWidget")
        CoordirateWidget.resize(1032, 871)
        self.toolButton = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton.setGeometry(QtCore.QRect(120, 50, 81, 101))
        self.toolButton.setObjectName("toolButton")
        self.toolButton_2 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_2.setGeometry(QtCore.QRect(670, 60, 81, 101))
        self.toolButton_2.setObjectName("toolButton_2")
        self.toolButton_3 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_3.setGeometry(QtCore.QRect(670, 220, 81, 101))
        self.toolButton_3.setObjectName("toolButton_3")
        self.toolButton_4 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_4.setGeometry(QtCore.QRect(120, 210, 81, 101))
        self.toolButton_4.setObjectName("toolButton_4")
        self.toolButton_5 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_5.setGeometry(QtCore.QRect(120, 570, 81, 101))
        self.toolButton_5.setObjectName("toolButton_5")
        self.toolButton_6 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_6.setGeometry(QtCore.QRect(120, 380, 81, 101))
        self.toolButton_6.setObjectName("toolButton_6")
        self.toolButton_7 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_7.setGeometry(QtCore.QRect(670, 390, 81, 101))
        self.toolButton_7.setObjectName("toolButton_7")
        self.toolButton_8 = QtWidgets.QToolButton(CoordirateWidget)
        self.toolButton_8.setGeometry(QtCore.QRect(670, 570, 81, 101))
        self.toolButton_8.setObjectName("toolButton_8")
        self.lineEditL1 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditL1.setGeometry(QtCore.QRect(210, 110, 151, 41))
        self.lineEditL1.setObjectName("lineEditL1")
        self.lineEditR1 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditR1.setGeometry(QtCore.QRect(760, 120, 151, 41))
        self.lineEditR1.setObjectName("lineEditR1")
        self.lineEditL2 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditL2.setGeometry(QtCore.QRect(210, 270, 151, 41))
        self.lineEditL2.setObjectName("lineEditL2")
        self.lineEditR2 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditR2.setGeometry(QtCore.QRect(760, 280, 151, 41))
        self.lineEditR2.setObjectName("lineEditR2")
        self.lineEditL3 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditL3.setGeometry(QtCore.QRect(210, 440, 151, 41))
        self.lineEditL3.setObjectName("lineEditL3")
        self.lineEditR3 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditR3.setGeometry(QtCore.QRect(760, 450, 151, 41))
        self.lineEditR3.setObjectName("lineEditR3")
        self.lineEditL4 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditL4.setGeometry(QtCore.QRect(210, 630, 151, 41))
        self.lineEditL4.setObjectName("lineEditL4")
        self.lineEditR4 = QtWidgets.QLineEdit(CoordirateWidget)
        self.lineEditR4.setGeometry(QtCore.QRect(760, 630, 151, 41))
        self.lineEditR4.setObjectName("lineEditR4")
        self.groupBox = QtWidgets.QGroupBox(CoordirateWidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 961, 731))
        self.groupBox.setObjectName("groupBox")
        self.save_coordirate_button = QtWidgets.QPushButton(CoordirateWidget)
        self.save_coordirate_button.setGeometry(QtCore.QRect(670, 790, 241, 51))
        self.save_coordirate_button.setObjectName("save_coordirate_button")

        self.retranslateUi(CoordirateWidget)
        QtCore.QMetaObject.connectSlotsByName(CoordirateWidget)

    def retranslateUi(self, CoordirateWidget):
        _translate = QtCore.QCoreApplication.translate
        CoordirateWidget.setWindowTitle(_translate("CoordirateWidget", "Form"))
        self.toolButton.setText(_translate("CoordirateWidget", "NO1_L"))
        self.toolButton_2.setText(_translate("CoordirateWidget", "NO1_R"))
        self.toolButton_3.setText(_translate("CoordirateWidget", "NO1_R"))
        self.toolButton_4.setText(_translate("CoordirateWidget", "NO1_L"))
        self.toolButton_5.setText(_translate("CoordirateWidget", "NO1_L"))
        self.toolButton_6.setText(_translate("CoordirateWidget", "NO1_L"))
        self.toolButton_7.setText(_translate("CoordirateWidget", "NO1_R"))
        self.toolButton_8.setText(_translate("CoordirateWidget", "NO1_R"))
        self.groupBox.setTitle(_translate("CoordirateWidget", "车间实景坐标"))
        self.save_coordirate_button.setText(_translate("CoordirateWidget", "保存"))

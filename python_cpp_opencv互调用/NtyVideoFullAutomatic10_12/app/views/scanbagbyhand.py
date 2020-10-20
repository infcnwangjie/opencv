import sys
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QMessageBox, QRadioButton, QPushButton, QVBoxLayout, \
	QTabWidget, QFormLayout, QLabel, QLineEdit, QSplitter, QSlider, QLCDNumber
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QIcon


class ScanBagByHandWindow(QWidget):
	def __init__(self, process):
		super().__init__()
		self.set_ui()
		self.init_button()
		self.process = process

	def set_ui(self):
		self.setWindowIcon(QIcon(":icons/shoudong.png"))
		self.setWindowTitle("手动扫描")
		self.manual_operation_layout = QGridLayout()
		self.shift_label = QLabel(self)
		self.shift_label.setText("设置距离")
		self.manual_operation_layout.addWidget(self.shift_label, *(0, 0, 1, 1))
		# 步数设置
		self.shift_edit = QLineEdit(self)

		self.manual_operation_layout.addWidget(self.shift_edit, *(0, 1, 1, 1))

		self.justscan_pushbutton = QPushButton(self)
		self.justscan_pushbutton.setText("静点扫描")
		self.manual_operation_layout.addWidget(self.justscan_pushbutton, *(1, 0, 1, 1))

		self.east_pushbutton = QPushButton(self)
		self.east_pushbutton.setText("向东")
		self.manual_operation_layout.addWidget(self.east_pushbutton, *(1, 1, 1, 1))
		self.west_pushbutton = QPushButton(self)
		self.west_pushbutton.setText("向西")
		self.manual_operation_layout.addWidget(self.west_pushbutton, *(1, 2, 1, 1))
		self.stop_pushbutton = QPushButton(self)
		self.stop_pushbutton.setText("停止扫描")
		self.manual_operation_layout.addWidget(self.stop_pushbutton, *(1, 3, 1, 1))

		# self.resize(400,200)
		self.setLayout(self.manual_operation_layout)

	def scan_east_direct(self):
		'''
		向东扫描
		'''
		reply = QMessageBox.information(self,  # 使用infomation信息框
		                                "友情提示",
		                                "确保没有向西传值",
		                                QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			shift_value = self.shift_edit.text()
			if shift_value is None or shift_value == "":
				return
			else:
				self.process.intelligentthread.scan_bag = True
				self.process.plchandle.move(east=int(shift_value))

	def scan_west_direct(self):
		'''
		向西扫描
		'''
		# print(self.shift_edit.text())
		reply = QMessageBox.information(self,  # 使用infomation信息框
		                                "友情提示",
		                                "确保没有向东传值",
		                                QMessageBox.Yes | QMessageBox.No)
		if reply == QMessageBox.Yes:
			shift_value = self.shift_edit.text()
			if shift_value is None or shift_value == "":
				return
			else:
				self.process.intelligentthread.scan_bag = True
				self.process.plchandle.move(west=int(shift_value))

	def stop_scan(self):
		'''
		停止扫描
		'''
		# print("停止扫描")
		self.process.intelligentthread.scan_bag = False
		self.process.plchandle.clear_plc()

	def just_scan(self):
		'''
		停止扫描
		'''
		# print("停止扫描")

		self.process.intelligentthread.scan_bag = True
		self.process.plchandle.clear_plc()
		# self.shift_edit.setEnabled(True)

	def init_button(self):
		self.justscan_pushbutton.clicked.connect(self.just_scan)
		self.east_pushbutton.clicked.connect(self.scan_east_direct)
		self.west_pushbutton.clicked.connect(self.scan_west_direct)
		self.stop_pushbutton.clicked.connect(self.stop_scan)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	demo = ScanBagByHandWindow()
	demo.show()
	sys.exit(app.exec_())

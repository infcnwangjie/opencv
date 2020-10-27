import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from app.icons import resource
class Table(QWidget):
	def __init__(self):
		super().__init__()
		self.init_ui()

	def init_ui(self):
		conLayout = QHBoxLayout()
		table = QTableWidget()
		table.setColumnCount(3)
		table.setRowCount(5)

		# table.setHorizontalHeaderLabels(['图片1', '图片2', '图片3'])

		table.setEditTriggers(QAbstractItemView.NoEditTriggers)

		table.setIconSize(QSize(300, 200));

		for i in range(3):  # 让列宽和图片相同
			table.setColumnWidth(i, 300)
		for i in range(5):  # 让行高和图片相同
			table.setRowHeight(i, 200)

		for k in range(15):  # 27 examples of DDA
			i = k / 3
			j = k % 3
			item = QTableWidgetItem()
			# item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中

			item.setIcon(QIcon(":icons/load.png"))

			table.setItem(i, j, item)

		conLayout.addWidget(table)
		self.setLayout(conLayout)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	example = Table()
	example.show()
	sys.exit(app.exec_())

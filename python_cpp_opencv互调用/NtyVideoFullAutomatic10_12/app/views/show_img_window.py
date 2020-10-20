import sys

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QRadioButton, QVBoxLayout, QTabWidget, QFormLayout, \
    QLabel, QLineEdit, QTextEdit, QListWidget, QListView, QTableWidget, QAbstractItemView, QTableWidgetItem
from PyQt5.uic.properties import QtCore, QtGui


class ShowWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.final_img_tab=QWidget()
        self.error_tab=QWidget()
        self.work_record_Tab=QWidget()

        self.final_img_tab_ui()
        self.error_tab_ui()
        self.workrecord_tab_ui()

        self.addTab(self.final_img_tab,'图像监控')
        self.addTab(self.error_tab,'故障追溯')
        self.addTab(self.work_record_Tab, '抓取记录')

    def final_img_tab_ui(self):
        layout = QVBoxLayout()
        self.final_picture_label = QLabel(self)
        self.final_picture_label.setObjectName("final_picture_label")
        self.final_picture_label.resize(900,700)
        layout.addWidget(self.final_picture_label)
        self.final_img_tab.setLayout(layout)

    def  error_tab_ui(self):
        layout = QVBoxLayout()
        rowcount,columncount=4,3
        self.tablewidget = QTableWidget()
        self.tablewidget.setColumnCount(columncount)
        self.tablewidget.setRowCount(rowcount)
        self.tablewidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablewidget.setIconSize(QSize(260, 300))
        for row in range(rowcount):  # 让行高和图片相同
            self.tablewidget.setRowHeight(row, 300)
        for col in range(columncount):  # 让列宽和图片相同
            self.tablewidget.setColumnWidth(col, 300)

        self.south_north_server = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.south_north_server.setIcon(QIcon(":icons/sifu.png"))
        # south_north_server.setSizeHint(QSize(300,300))
        self.south_north_server.setText("南北伺服")
        self.tablewidget.setItem(0, 0, self.south_north_server)

        self.east_server = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.east_server.setIcon(QIcon(":icons/sifu.png"))
        # east_server.setSizeHint(QSize(300, 300))
        self.east_server.setText("东伺服")
        self.tablewidget.setItem(0, 1, self.east_server)

        self.west_server = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.west_server.setIcon(QIcon(":icons/sifu.png"))
        # west_server.setSizeHint(QSize(300, 300))
        self.west_server.setText("西伺服")
        self.tablewidget.setItem(0, 2, self.west_server)

        self. south_xianwei = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.south_xianwei.setIcon(QIcon(":icons/xianweiqi.png"))
        # south_xianwei.setSizeHint(QSize(300, 300))
        self.south_xianwei.setText("南限位")
        self.tablewidget.setItem(1, 0, self.south_xianwei)

        self.north_xianwei = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.north_xianwei.setIcon(QIcon(":icons/xianweiqi.png"))
        # north_xianwei.setSizeHint(QSize(300, 300))
        self.north_xianwei.setText("北限位")
        self.tablewidget.setItem(1, 1, self.north_xianwei)


        self.east_xianwei = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.east_xianwei.setIcon(QIcon(":icons/xianweiqi.png"))
        # east_xianwei.setSizeHint(QSize(300, 300))
        self.east_xianwei.setText("东限位")
        self.tablewidget.setItem(1, 2, self.east_xianwei)

        self.west_xianwei = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.west_xianwei.setIcon(QIcon(":icons/xianweiqi.png"))
        # west_xianwei.setSizeHint(QSize(300, 300))
        self.west_xianwei.setText("西限位")
        self.tablewidget.setItem(2, 0, self.west_xianwei)


        self.south_north_server_trip = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.south_north_server_trip.setIcon(QIcon(":icons/dianzha.png"))
        # south_north_server_trip.setSizeHint(QSize(300, 300))
        self.south_north_server_trip.setText("南北伺服断电情况")
        self.tablewidget.setItem(2, 1, self.south_north_server_trip)

        self.eastwest_server1_trip = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.eastwest_server1_trip.setIcon(QIcon(":icons/dianzha.png"))
        # eastwest_server1_trip.setSizeHint(QSize(300, 300))
        self.eastwest_server1_trip.setText("东西伺服1断电情况")
        self.tablewidget.setItem(2, 2, self.eastwest_server1_trip)

        self.eastwest_server2_trip = QTableWidgetItem()
        # item.setFlags(ItemIsEnabled)  # 用户点击时表格时，图片被选中
        self.eastwest_server2_trip.setIcon(QIcon(":icons/dianzha.png"))
        # eastwest_server2_trip.setSizeHint(QSize(300, 300))
        self.eastwest_server2_trip.setText("东西伺服2断电情况")
        self.tablewidget.setItem(3, 0, self.eastwest_server2_trip)

        # self.text = QTextEdit()
        # self.text.setFontWeight(18)
        layout.addWidget(self.tablewidget)
        # layout.addWidget(self.text)

        self.error_tab.setLayout(layout)

    def  workrecord_tab_ui(self):
        layout = QVBoxLayout()
        self.workrecord = QTextEdit()
        self.workrecord.setStyleSheet("""
					QTextEdit:
					{
					border: 1px solid yellow;
					# background:qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 yellow, stop:1 yellow);
					border-radius: 3px;
					# height: 8px;
					font-size:18px;
					}
				""")
        layout.addWidget(self.workrecord)
        self.work_record_Tab.setLayout(layout)


    



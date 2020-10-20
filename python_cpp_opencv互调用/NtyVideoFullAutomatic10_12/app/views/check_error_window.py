import sys
from PyQt5.QtWidgets import QWidget,QApplication,QGridLayout,QRadioButton,QVBoxLayout,QTabWidget,QFormLayout,QLabel,QLineEdit

class CheckErrorWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.tab1=QWidget()
        self.tab2=QWidget()
        self.tab3=QWidget()

        self.addTab(self.tab1,'南北伺服')
        self.addTab(self.tab2,'东伺服')
        self.addTab(self.tab3,'西伺服')

        self.NorthSourthUI()
        self.EastUI()
        self.WestUI()

    def NorthSourthUI(self):
        tab1_lay=QGridLayout()
        self.tab1.setLayout(tab1_lay)

        self.setTabText(0,'南北伺服')
        server_warn_label=QLabel('南北伺服报警：')
        self.northsourth_server_warn_edit=QLineEdit()

        south_limit_warn_label=QLabel('南限位报警：')
        self.south_limit_warn_edit=QLineEdit()

        north_limit_warn_label=QLabel('北限位报警：')
        self.north_limit_warn_edit=QLineEdit()


        tab1_lay.addWidget(server_warn_label,0,0)
        tab1_lay.addWidget(self.northsourth_server_warn_edit,0,1)

        tab1_lay.addWidget(south_limit_warn_label,1,0)
        tab1_lay.addWidget(self.south_limit_warn_edit,1,1)

        tab1_lay.addWidget(north_limit_warn_label,2,0)
        tab1_lay.addWidget(self.north_limit_warn_edit,2,1)


    def EastUI(self):
        tab2_lay=QGridLayout()
        self.tab2.setLayout(tab2_lay)

        self.setTabText(1,'东伺服')

        east_server_warn_label=QLabel('东伺服报警：')
        self.east_server_warn_edit=QLineEdit()

        east_limit_warn_label=QLabel('东限位报警：')
        self.east_limit_warn_edit=QLineEdit()

        tab2_lay.addWidget(east_server_warn_label,0,0)
        tab2_lay.addWidget(self.east_server_warn_edit,0,1)

        tab2_lay.addWidget(east_limit_warn_label,1,0)
        tab2_lay.addWidget(self.east_limit_warn_edit,1,1)

      


    def WestUI(self):
        tab3_lay=QGridLayout()
        self.tab3.setLayout(tab3_lay)

        west_server_warn_label=QLabel('西伺服报警：')
        self.west_server_warn_edit=QLineEdit()

        west_limit_warn_label=QLabel('西限位报警：')
        self.west_limit_warn_edit=QLineEdit()

        tab3_lay.addWidget(west_server_warn_label,0,0)
        tab3_lay.addWidget(self.west_server_warn_edit,0,1)

        tab3_lay.addWidget(west_limit_warn_label,1,0)
        tab3_lay.addWidget(self.west_limit_warn_edit,1,1)

    def  write_error_message(self,error_info:dict):
        if 'south_north' in error_info:
            SOUTH_NORTH_SERVER_FLAG=error_info['south_north']
            self.northsourth_server_warn_edit.setText('南北伺服报警' if SOUTH_NORTH_SERVER_FLAG is not None and SOUTH_NORTH_SERVER_FLAG==1 else "南北伺服正常")
            if 'south' in error_info:
                SOUTH_LIMIT_FLAG=error_info['south']
                self.south_limit_warn_edit.setText( '南限位报警' if  SOUTH_LIMIT_FLAG is not None and SOUTH_LIMIT_FLAG==1 else '南限位正常')
            if 'north' in error_info:
                NORTH_LIMIT_FLAG=error_info['north']
                self.north_limit_warn_edit.setText('北限位报警' if NORTH_LIMIT_FLAG is not None and NORTH_LIMIT_FLAG==1 else '北限位正常')
        
        if 'east_server' in error_info:
            EAST_SERVER_FLAG=error_info['east_server']
            self.east_server_warn_edit.setText('东伺服报警' if EAST_SERVER_FLAG is not None and EAST_SERVER_FLAG==1 else '东伺服正常')
            if 'east' in error_info:
                EAST_LIMIT_FLAG=error_info['east']
                self.east_limit_warn_edit.setText('东限位报警' if EAST_LIMIT_FLAG is not None and EAST_LIMIT_FLAG==1 else '东限位正常')

        if 'west_server' in error_info:
            WEST_SERVER_FLAG=error_info['west_server']
            self.west_server_warn_edit.setText('西伺服报警' if WEST_SERVER_FLAG is not None and WEST_SERVER_FLAG==1 else '西伺服正常')
            if 'west' in error_info:
                WEST_LIMIT_FLAG=error_info['west']
                self.west_limit_warn_edit.setText('西限位报警' if WEST_LIMIT_FLAG is not None and WEST_LIMIT_FLAG==1 else '西限位正常')
        


class TabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('demo')
        self.setGeometry(300,300,300,200)
        layout=QFormLayout()
        self.setLayout(layout)
        tabwidget=CheckErrorWidget()
        layout.addRow(tabwidget)
    



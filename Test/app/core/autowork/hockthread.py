from PyQt5.QtCore import QThread, pyqtSignal

from app.core.plc.plchandle import PlcHandle
from app.status import HockStatus


class HockThread(QThread):
    '''钩子运动的时候，一定要向PLC中写入X，Y位置参数'''
    askforSingnal = pyqtSignal(str)
    moveSignal = pyqtSignal(str)

    def __init__(self, plchandle: PlcHandle):
        super().__init__(parent=None)
        self._movex = None
        self._movey = None
        self._on = True
        self.plchandle = plchandle

    @property
    def work(self):
        return self._on

    @work.setter
    def work(self, value=True):
        self._on = value

    @property
    def x(self):
        return self._movex

    @x.setter
    def x(self, value):
        self._movex = value

    @property
    def y(self):
        return self._movey

    @y.setter
    def y(self, value):
        self._movey = value

    def run(self):
        if self.on == True:
            plcstatus = self.plchandle.read_status()
            # 'move', 'stop', 'reback'
            if plcstatus in ['stop', 'reback']:
                self.askforSingnal.emit("plc need hock  position!")
            elif plcstatus == "move":
                self.moveSignal.emit("hock moveing")

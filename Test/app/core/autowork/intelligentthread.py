from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from app.core.exceptions.allexception import SdkException, NotFoundBagException, NotFoundHockException
from app.core.plc.plchandle import PlcHandle
from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK
from app.status import HockStatus


class IntelligentThread(QThread):
    positionSignal = pyqtSignal(tuple)  # 已注册
    dropHockSignal = pyqtSignal(float)
    dropBagSignal = pyqtSignal(float)
    rebackSignal = pyqtSignal(str)
    finishSignal = pyqtSignal(str)
    foundbagSignal=pyqtSignal(int)

    def __init__(self, video_player, IMGHANDLE=None,
                 positionservice: PointLocationService = None, parent=None):
        super().__init__(parent=parent)
        self._playing = True
        self._finish = False
        self._working = False
        self.video_player = video_player
        self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
        self.positionservice = positionservice  # 指令处理器
        self._hockstatus = HockStatus.POSITION  # 钩子状态会影响定位程序
        self.firstfoundbag=True

    def __del__(self):
        self._working = False
        if hasattr(self.IMAGE_HANDLE, 'release') and self.IMAGE_HANDLE:
            self.IMAGE_HANDLE.release()

    @property
    def play(self):
        return self._playing

    @play.setter
    def play(self, value=True):
        self._playing = value

    @property
    def work(self):
        return self._working

    @work.setter
    def work(self, value=True):
        self._working = value

    @property
    def hockstatus(self):
        return self._hockstatus

    @hockstatus.setter
    def hockstatus(self, value=HockStatus.POSITION):
        # value must HockStatus enum
        self._hockstatus = value

    def run(self):

        while self.play:
            sleep(1 / 22)
            frame = self.IMAGE_HANDLE.read()
            if frame is None:
                self.sinOut.emit("frame is None！")
                self.finish = True
                self.sinOut.emit("操作完毕")
                break
            frame = cv2.resize(frame, (800, 800))

            if frame.ndim == 3:
                show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.ndim == 2:
                show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if self.work:
                self.positionservice.img = show
                self.process()
                show = self.positionservice.img
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.video_player.setPixmap(QPixmap.fromImage(showImage))
            self.video_player.setScaledContents(True)

    def process(self):

        if self.hockstatus == HockStatus.POSITION:
            try:
                location_info = self.positionservice.computelocations()
                if location_info is None:
                    return None
                img_distance, real_distance, real_x_distance, real_y_distance = location_info
                bagnum = len(self.positionservice.bags)
                if self.firstfoundbag:
                    self.foundbagSignal.emit(bagnum)
                    self.firstfoundbag=False
                self.nearest_bag = self.positionservice.nearestbag
                self.hock = self.positionservice.hock
                # print("向PLC中写入需要移动的X、Y轴移动距离")
                movex = real_x_distance
                movey = real_y_distance
                if 0 < real_x_distance < 10 and 0 < movey < 10:
                    self.positionSignal.emit((0, 0, -7))  # 暂时写死，钩子找到最近袋子，向下抛7米吧
                else:
                    self.positionSignal.emit((movex, movey, 0))  # Z轴不变
            except NotFoundBagException:
                self.finishSignal.emit("not found bag ,maybe finish")
            except NotFoundHockException:
                self.finishSignal.emit("not found hock ,maybe finish")

        elif self.hockstatus == HockStatus.DROP_HOCK:
            print("图像检测是否钩住袋子")
            self.positionservice.compute_hook_location()
            self.positionSignal.emit((0, 0, 7))  # 向上拉袋子
        elif self.hockstatus == HockStatus.PULL_HOCK:
            pass
        elif self.hockstatus == HockStatus.DROP_BAG:
            print("检测放置袋子区域")
            self.positionSignal.emit((0, -9, 0))  # 向放置区域移动
            distance_place = 0
            if distance_place == 0:
                self.rebackSignal.emit("reback")

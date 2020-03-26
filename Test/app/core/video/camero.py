from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class CameroThread(QThread):
    sinOut = pyqtSignal(str)

    def __init__(self, video_player, video_file=None, parent=None):
        super().__init__(parent=parent)
        self.working = True
        self.video_player = video_player
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
        else:
            self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.working = False
        self.cap.release()

    def run(self):
        while self.working:
            sleep(1/23)
            ret, frame = self.cap.read()
            if frame is None:
                self.sinOut.emit("frame is None！")
                break

            if frame.ndim == 3:
                show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.ndim == 2:
                show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

            # self.label.setPixmap(QPixmap.fromImage(showImage))

            # temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            self.video_player.setPixmap(QPixmap.fromImage(showImage))
            self.video_player.setScaledContents(True)
            if not self.working:
                self.sinOut.emit("被中断！")
                break

        self.sinOut.emit("操作完毕")

    def stop(self):
        self.working = False



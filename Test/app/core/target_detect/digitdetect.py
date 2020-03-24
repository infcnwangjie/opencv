import os

import cv2
import numpy as np

from app.config import TRAIN_DATA_DIR


class DigitDetector:
    '''用于从图像中检测数字:数字6和8需要特别注意'''
    def __init__(self, generalsamples_path=None, generalresponses_path=None):
        self.model = None
        self.generalsamples_path, self.generalresponses_path = generalsamples_path, generalresponses_path

    # 训练模型
    def practise(self):
        current_root_path = os.getcwd()
        if self.generalsamples_path is None:
            self.generalsamples_path = os.path.join(TRAIN_DATA_DIR, 'generalsamples.data')
        if self.generalresponses_path is None:
            self.generalresponses_path = os.path.join(TRAIN_DATA_DIR, 'generalresponses.data')
        try:
            samples = np.loadtxt(self.generalsamples_path, np.float32)
            responses = np.loadtxt(self.generalresponses_path, np.float32)
            responses = responses.reshape((responses.size, 1))
        except IOError as e:
            raise Exception("文件找不到")
        else:
            self.model = cv2.ml.KNearest_create()
            self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 用感兴趣的区域去匹配训练模型
    def fine_nearest(self, imgroi, k=1):
        retval, results, neigh_resp, dists = self.model.findNearest(imgroi, k=k)
        return retval, results, neigh_resp, dists

    def readnum(self, roi):
        roismall = cv2.resize(roi, (30, 30))
        roismall = roismall.reshape((1, 900))
        roismall = np.float32(roismall)
        _retval, results, _neigh_resp, _dists = self.fine_nearest(roismall, k=1)
        return results

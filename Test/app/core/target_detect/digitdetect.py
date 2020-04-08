# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np


class DigitDetector:
    '''用于从图像中检测数字:数字6和8需要特别注意'''

    def __init__(self):
        self.model = None
        self.practise()

    # 训练模型
    def practise(self):
        # current_root_path = os.getcwd()
        generalsamples_path = 'c:/work/generalsamples.data'
        generalresponses_path = 'c:/work/generalresponses.data'
        try:
            samples = np.loadtxt(generalsamples_path, np.float32)
            responses = np.loadtxt(generalresponses_path, np.float32)
            responses = responses.reshape((responses.size, 1))
        except IOError as e:
            raise Exception("文件找不到")
        else:
            self.model = cv2.ml.KNearest_create()
            self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 用感兴趣的区域去匹配训练模型 imgroi 为图像
    def fine_nearest(self, imgroi, k=1):
        retval, results, neigh_resp, dists = self.model.findNearest(imgroi, k=k)
        return retval, results, neigh_resp, dists

    def readnum(self, roi):
        # print(roi.shape)
        roismall = cv2.resize(roi, (30, 30))
        roismall = roismall.reshape((1, 900))
        # roismall = cv2.resize(roi, (60, 60))
        # roismall = roismall.reshape((1, 10000))
        roismall = np.float32(roismall)
        _retval, results, _neigh_resp, _dists = self.fine_nearest(roismall, k=1)
        return results

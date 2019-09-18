# -*- coding: utf-8 -*-
# encoding:utf-8
import os

import cv2 as cv
import re
import numpy as np
from PIL import Image


def detect_eye(img=None, filename=None, test=False):
    eye_cascade = cv.CascadeClassifier("./cascades/haarcascade_eye.xml")
    face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    if img is None:
        img = cv.imread(filename)
    # resize_img = cv.resize(img, (400, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
        for (ex, ey, ew, eh) in eyes:
            img = cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    if test:
        cv.namedWindow("test")
        cv.imshow("test", img)
        cv.waitKey(0)
    else:
        return img


def detect_face(img=None, filename=None, test=False):
    face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    if img is None:
        img = cv.imread(filename)
    # resize_img = cv.resize(img, (400, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if test:
        cv.namedWindow("test")
        cv.imshow("test", img)
        cv.waitKey(0)
    else:
        return img


def getTrainImg(filepath=None):
    X, Y = [], []
    pattern = re.compile("[0-9]+")
    for item in os.listdir(filepath):
        # print(item)
        im = cv.imread(os.path.join(filepath, item), cv.IMREAD_GRAYSCALE)
        im = cv.resize(im, (200, 200))
        X.append(np.asarray(im, dtype=np.uint8))
        label_mach = re.match(pattern, item)
        Y.append(label_mach.group(0))
    return [X, Y]


face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
names = {"1000": 'wangjie', "1001": 'wangjie', "1002": 'wangjie', "1003": 'wangjie', "1004": 'wangjie',
         "1005": 'wangjie', "1006": 'wangjie', "1007": 'wangjie',
         '2001': 'zhakeboge', '2002': 'zhakeboge', '2003': 'zhakeboge', '3001': 'yichaoshuo', '3002': 'yichaoshuo',
         '3003': 'yichaoshuo',
         '3004': 'yichaoshuo', '3005': 'yichaoshuo', '3006': 'yichaoshuo', '3007': 'yichaoshuo',
         '4001':'yibaoquan','4002':'yibaoquan','4003':'yibaoquan','5001':'yuemu','5002':'yuemu'}

position = {"wangjie": "当前位置西地华府", "yichaoshuo": "当前位置东华园", "zhakeboge": "当前位置纽约"}
[X, Y] = getTrainImg("./test")
Y = np.asarray(Y, dtype=np.int32)
model = cv.face.EigenFaceRecognizer_create()
model.train(X, Y)


def face_rec(img=None, filename=None, test=False):
    global face_cascade, model
    if img is None and filename is not None:
        img = cv.imread(filename)

    # resize_img = cv.resize(img, (400, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[x:x + w, y:y + h]
        if roi is None:
            break
        # print(roi.shape)
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return img
        roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_LINEAR)
        params = model.predict(roi)
        print(params)
        # cv.putText(img, names[str(params[0])], (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv.putText(img, names[str(params[0])], (x, y - 20), cv.FONT_HERSHEY_SIMPLEX,
                   1, 255, 2)
        # print(params)

    if test:
        cv.namedWindow("test")
        cv.imshow("test", img)
        cv.waitKey(0)
    else:
        return img


if __name__ == '__main__':
    # filename = "zhakeboge.jpg"
    # detect_face(filename=filename, test=True)
    # detect_eye(filename=filename, test=True)
    # face_rec(filename=filename, test=True)
    print('大胖子啊')
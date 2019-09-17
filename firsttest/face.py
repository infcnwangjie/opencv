# coding:utf-8
import os

import cv2 as cv
import re

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
    pattern = re.compile("[A-Za-z]+")
    for item in os.listdir(filepath):
        # print(item)
        im = cv.imread(os.path.join(filepath, item), cv.IMREAD_GRAYSCALE)
        X.append(im)
        label_mach = re.match(pattern, item)
        Y.append(label_mach.group(0))
    return X,Y


def face_rec(img=None, filename=None, test=False):
    names=['wangjie','zhakeboge']
    face_cascade = cv.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")


    if img is None:
        img = cv.imread(filename)
    # resize_img = cv.resize(img, (400, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    model=cv.face.EigenFaceRecognizer_create()
    model.train()

    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if test:
        cv.namedWindow("test")
        cv.imshow("test", img)
        cv.waitKey(0)
    else:
        return img


if __name__ == '__main__':
    filename = "zhakeboge.jpg"
    # detect_face(filename=filename, test=True)
    # detect_eye(filename=filename, test=True)
    getTrainImg(filepath="./test")

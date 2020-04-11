import os

import cv2
import numpy as np


for imgname in os.listdir(r"E:\darknet-master\scripts\VOCdevkit\VOC2020\JPEGImages\src"):
    img=cv2.imread("E:/darknet-master/scripts/VOCdevkit/VOC2020/JPEGImages/src/{}".format(imgname))
    # print("已处理" + imgname)
    process_img = cv2.resize(img, (608,608))
    cv2.imwrite("E:/darknet-master/scripts/VOCdevkit/VOC2020/JPEGImages/dest/"+imgname,process_img)
    print("已处理"+imgname)

print("finish")
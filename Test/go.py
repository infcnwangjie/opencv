import ctypes
import os
from ctypes import cdll, c_uint, c_void_p, c_int, c_float, c_char_p,POINTER,byref,Structure,cast,c_uint8

import cv2
import numpy as np

OPENCV_SUPPLYDLL = cdll.LoadLibrary("C:/work/cpp/build-OPENCV_SUPPLY-Desktop_Qt_5_14_2_MinGW_64_bit-Debug/libOPENCV_SUPPLY.dll")

print(OPENCV_SUPPLYDLL.helloWorld())
print(OPENCV_SUPPLYDLL.add(56,78))


img = cv2.imread('c:/work/nty/hangche/2020-05-15-15-59-16test.bmp')
h, w, c = img.shape[0], img.shape[1], img.shape[2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv",hsv)
src_target = np.asarray(hsv, dtype=np.uint8)
src_target = src_target.ctypes.data_as(ctypes.c_char_p)

OPENCV_SUPPLYDLL.find_it.restype =ctypes.POINTER(ctypes.c_uint8)
# OPENCV_SUPPLYDLL.find_it.argtype = [POINTER(IMAGE), POINTER(IMAGE)]
# # gray = np.reshape(hsv, (h, w, 3))          # 一定要使用(h, w, 1)，最后的1别忘。
# im = array_to_image(hsv)
# # gray_img = array_to_image(gray)
# #
roiimg = cv2.imread('C:/NTY_IMG_PROCESS/ROIS/NO1_L.png')
# cv2.imshow("roiimg",roiimg)
roihsv = cv2.cvtColor(roiimg, cv2.COLOR_BGR2HSV)
# model_img = array_to_image(roiimg)
model = np.asarray(roihsv, dtype=np.uint8)
model = model.ctypes.data_as(ctypes.c_char_p)
model_h, model_w, model_c = roiimg.shape[0], roiimg.shape[1], roiimg.shape[2]

pointer=OPENCV_SUPPLYDLL.find_it(src_target,model,w,h,model_w,model_h)
print(pointer)
# # #测试strTest方法
foreground = np.array(np.fromiter(pointer, dtype=np.uint8, count=h * w))
result=foreground.reshape((h,w))
cv2.imshow("img",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
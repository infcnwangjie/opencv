import os
from ctypes import cdll, c_uint, c_void_p, c_int, c_float, c_char_p,POINTER,byref,Structure,cast,c_uint8

import cv2
import numpy as np

#调用库

MWORKDLL = cdll.LoadLibrary("c:/libfind_roi.dll")


def c_array(ctype, values):  # 把图像的数据转化为内存连续的列表使c++能使用这块内存
	arr = (ctype * len(values))()
	arr[:] = values
	return arr


def array_to_image(arr):
	c = arr.shape[2]
	h = arr.shape[0]
	w = arr.shape[1]
	arr = arr.flatten()
	data = c_array(c_uint8, arr)
	im = IMAGE(w, h, c, data)
	return im


class IMAGE(Structure):  # 这里和ImgSegmentation.hpp里面的结构体保持一致。
	_fields_ = [("w", c_int),
				("h", c_int),
				("c", c_int),
				("data", POINTER(c_uint8))]



img = cv2.imread('c:/work/nty/hangche/2020-05-15-15-59-16test.bmp')
h, w, c = img.shape[0], img.shape[1], img.shape[2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# gray = np.reshape(hsv, (h, w, 3))          # 一定要使用(h, w, 1)，最后的1别忘。
im = array_to_image(img)
# gray_img = array_to_image(gray)

roiimg = cv2.imread('C:/NTY_IMG_PROCESS/ROIS/NO1_L.png')
model_img = array_to_image(roiimg)


#测试strTest方法
MWORKDLL.find_object.argtype = [POINTER(IMAGE), POINTER(IMAGE)]
MWORKDLL.find_object.restype = POINTER(IMAGE)
c=MWORKDLL.find_object(im,model_img)
cv2.imshow(c)
cv2.waitKey(c)
cv2.destroyAllWindows()
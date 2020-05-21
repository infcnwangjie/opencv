import ctypes
import os
from ctypes import cdll, c_uint, c_void_p, c_int, c_float, c_char_p, POINTER, byref, Structure, cast, c_uint8

import cv2
import numpy as np

OPENCV_SUPPLYDLL = cdll.LoadLibrary(
    "C:/NTY_IMG_PROCESS/dll/libOPENCV_SUPPLY.dll")

print(OPENCV_SUPPLYDLL.helloWorld())
print(OPENCV_SUPPLYDLL.add(56, 78))


def cpp_canny(input):
    if len(img.shape) >= 3 and img.shape[-1] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[0], gray.shape[1]

    # 获取numpy对象的数据指针
    frame_data = np.asarray(gray, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)

    # 设置输出数据类型为uint8的指针
    OPENCV_SUPPLYDLL.cpp_canny.restype = ctypes.POINTER(ctypes.c_uint8)

    # 调用dll里的cpp_canny函数
    pointer = OPENCV_SUPPLYDLL.cpp_canny(h, w, frame_data)

    # 从指针指向的地址中读取数据，并转为numpy array
    np_canny = np.array(np.fromiter(pointer, dtype=np.uint8, count=h * w))

    return pointer, np_canny.reshape((h, w))


img = cv2.imread('c:/work/nty/hangche/2020-05-15-15-59-16test.bmp')
img = cv2.resize(img, (700, 900))
ptr, canny = cpp_canny(img)
cv2.imshow('canny', canny)
cv2.waitKey(2000)


def find_it(input, model):

    input_h, input_w = input.shape[0], input.shape[1]
    m_h, m_w = model.shape[0], model.shape[1]

    # 获取numpy对象的数据指针
    frame_data = np.asarray(input, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)

    model_data = np.asarray(model, dtype=np.uint8)
    model_data = model_data.ctypes.data_as(ctypes.c_char_p)

    # 设置输出数据类型为uint8的指针
    OPENCV_SUPPLYDLL.find_it.restype = ctypes.POINTER(ctypes.c_uint8)

    # 调用dll里的cpp_canny函数
    pointer = OPENCV_SUPPLYDLL.find_it(frame_data,model_data,input_w,input_h,m_w,m_h)

    # 从指针指向的地址中读取数据，并转为numpy array
    result_img = np.array(np.fromiter(pointer, dtype=np.uint8, count=input_h * input_w))

    return result_img.reshape((input_h, input_w))


input = cv2.imread('c:/work/nty/hangche/2020-05-15-15-59-16test.bmp')
input = cv2.resize(input, (700, 900))

model = cv2.imread('C:/NTY_IMG_PROCESS/ROIS/NO1_L.png')
m_h, m_w, m_c = model.shape[0], model.shape[1], model.shape[2]
foreground = find_it(input, model)
cv2.imshow('foreground', foreground)
cv2.waitKey(2000)

cv2.destroyAllWindows()




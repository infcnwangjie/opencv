import ctypes
from ctypes import *
import numpy as np

from app.config import *


# print(PLAT)


class MvSuply:
    @staticmethod
    # @cost_time
    def FIND_IT( input, model):
        OPENCV_SUPPLYDLL = cdll.LoadLibrary(
            SUPPLY_OPENCV_DLL_64_PATH if PLAT == '64' else SUPPLY_OPENCV_DLL_32_PATH)
        input_h, input_w = input.shape[0], input.shape[1]
        m_h, m_w = model.shape[0], model.shape[1]

        OPENCV_SUPPLYDLL.find_it.restype = ctypes.POINTER(ctypes.c_uint8)
        result_img = np.array(
            np.fromiter(OPENCV_SUPPLYDLL.find_it(np.array(input, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
                                                     np.asarray(model, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
                                                     input_w, input_h, m_w, m_h),
                        dtype=np.uint8, count=input_h * input_w))
        return result_img.reshape((input_h, input_w))

    @staticmethod
    def SAME_RATE(IMG1, IMG2):
        OPENCV_SUPPLYDLL = cdll.LoadLibrary(
            SUPPLY_OPENCV_DLL_64_PATH if PLAT == '64' else SUPPLY_OPENCV_DLL_32_PATH)
        try:
            img1_h, img1_w = IMG1.shape[0], IMG1.shape[1]
            img2_h, img2_w = IMG2.shape[0], IMG2.shape[1]

            RATE = OPENCV_SUPPLYDLL.same_rate(np.array(IMG1, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
                                                np.asarray(IMG2, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
                                                img1_w, img1_h, img2_w, img2_h)
        except Exception as e:
            print(e.__str__())
            return 0
        return RATE


    @staticmethod
    # @cost_time
    def CATEGORY_CODE( TEST_IMG):

        OPENCV_SUPPLYDLL = cdll.LoadLibrary(
            SUPPLY_OPENCV_DLL_64_PATH if PLAT == '64' else SUPPLY_OPENCV_DLL_32_PATH)

        try:
            img1_h, img1_w = TEST_IMG.shape[0], TEST_IMG.shape[1]

            result = OPENCV_SUPPLYDLL.category_code(
                np.array(TEST_IMG, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
                img1_h,
                img1_w
                )
        except:
            return 0

        return result

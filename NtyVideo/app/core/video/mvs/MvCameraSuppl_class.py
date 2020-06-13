import ctypes
from ctypes import *
import numpy as np

from app.config import *

OPENCV_SUPPLYDLL = cdll.LoadLibrary(
	SUPPLY_OPENCV_DLL_64_PATH if PLAT == '64' else SUPPLY_OPENCV_DLL_32_PATH)


class MvSuply:
	# 获取特征值
	@staticmethod
	def FIND_IT(input, model):
		input_h, input_w = input.shape[0], input.shape[1]
		m_h, m_w = model.shape[0], model.shape[1]

		OPENCV_SUPPLYDLL.find_it.restype = ctypes.POINTER(ctypes.c_uint8)
		result_img = np.array(
			np.fromiter(OPENCV_SUPPLYDLL.find_it(np.array(input, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
			                                     np.asarray(model, dtype=np.uint8).ctypes.data_as(ctypes.c_char_p),
			                                     input_w, input_h, m_w, m_h),
			            dtype=np.uint8, count=input_h * input_w))
		return result_img.reshape((input_h, input_w))

# -- coding: utf-8 --

import sys
import copy
import msvcrt

from ctypes import *
import time
import numpy as np
import cv2

from app.core.exceptions.allexception import SdkException
from app.core.video.mvs.CameraParams_const import MV_GIGE_DEVICE, MV_USB_DEVICE, MV_ACCESS_Exclusive
from app.core.video.mvs.CameraParams_header import MV_CC_DEVICE_INFO_LIST, MV_TRIGGER_MODE_OFF, MVCC_INTVALUE, \
	MV_FRAME_OUT_INFO_EX, MV_SAVE_IMAGE_PARAM_EX, MV_Image_Bmp, MV_CC_DEVICE_INFO
from app.core.video.mvs.MvCameraControl_class import MvCamera


class SdkHandle(object):
	'''海康sdk句柄'''
	def __init__(self):
		self.deviceList = MV_CC_DEVICE_INFO_LIST()
		self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
		self.image = None
		# ch:创建相机实例 | en:Creat Camera Object
		self.cam = None
		self.stParam = None  # 数据包信息
		self.init_all()

	# # ch:枚举设备 | en:Enum device
	def enum_devices(self):
		# ch:枚举设备 | en:Enum device
		ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
		if ret != 0:
			print("enum devices fail! ret[0x%x]" % ret)
			raise SdkException("enum devices fail! ret[0x%x]" % ret)

		if self.deviceList.nDeviceNum == 0:
			print("find no device!")
			raise SdkException("find no device!")

		print("find %d devices!" % self.deviceList.nDeviceNum)

		for i in range(0, self.deviceList.nDeviceNum):
			mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
			if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
				print("\ngige device: [%d]" % i)
				strModeName = ""
				for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
					strModeName = strModeName + chr(per)
				print("device model name: %s" % strModeName)

				nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
				nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
				nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
				nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
				print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
			elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
				print("\nu3v device: [%d]" % i)
				strModeName = ""
				for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
					if per == 0:
						break
					strModeName = strModeName + chr(per)
				print("device model name: %s" % strModeName)

				strSerialNumber = ""
				for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
					if per == 0:
						break
					strSerialNumber = strSerialNumber + chr(per)
				print("user serial number: %s" % strSerialNumber)

	# 开启工业相机
	def create_camera(self):
		self.cam = MvCamera()
		# ch:选择设备并创建句柄 | en:Select device and create handle
		stDeviceList = cast(self.deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

		ret = self.cam.MV_CC_CreateHandle(stDeviceList)
		if ret != 0:
			print("create handle fail! ret[0x%x]" % ret)

		# ch:打开设备 | en:Open device
		ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
		if ret != 0:
			print("open device fail! ret[0x%x]" % ret)

		# ch:探测网络最f佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
		if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
			nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
			if int(nPacketSize) > 0:
				ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
				if ret != 0:
					print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
			else:
				print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

		# ch:设置触发模式为off | en:Set trigger mode as off
		ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
		if ret != 0:
			print("set trigger mode fail! ret[0x%x]" % ret)

		# ch:获取数据包大小 | en:Get payload size
		self.stParam = MVCC_INTVALUE()
		memset(byref(self.stParam), 0, sizeof(MVCC_INTVALUE))

		ret = self.cam.MV_CC_GetIntValue("PayloadSize", self.stParam)
		if ret != 0:
			print("get payload size fail! ret[0x%x]" % ret)
			sys.exit()

		ret = self.cam.MV_CC_StartGrabbing()
		if ret != 0:
			print("start grabbing fail! ret[0x%x]" % ret)
			sys.exit()

	def init_all(self):
		self.enum_devices()  # ch:枚举设备 | en:Enum device
		self.create_camera()  # 创建相机

	def read(self):
		nPayloadSize = self.stParam.nCurValue
		stDeviceList = MV_FRAME_OUT_INFO_EX()
		memset(byref(stDeviceList), 0, sizeof(stDeviceList))
		data_buf = (c_ubyte * nPayloadSize)()
		ret = self.cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 1000)
		if ret == 0:
			# Stop = time()
			# print(Stop - start)
			print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
				stDeviceList.nWidth, stDeviceList.nHeight, stDeviceList.nFrameNum))

			stConvertParam = MV_SAVE_IMAGE_PARAM_EX()
			stConvertParam.nWidth = stDeviceList.nWidth
			stConvertParam.nHeight = stDeviceList.nHeight
			stConvertParam.pData = data_buf
			stConvertParam.nDataLen = stDeviceList.nFrameLen
			stConvertParam.enPixelType = stDeviceList.enPixelType

			# MV_Image_Undefined  = 0, //未定义
			#   MV_Image_Bmp        = 1, //BMP图片
			#   MV_Image_Jpeg       = 2, //JPEG图片
			#   MV_Image_Png        = 3, //PNG图片，暂不支持
			#   MV_Image_Tif        = 4, //TIF图片，暂不支持

			# jpg参数
			# stConvertParam.nJpgQuality   = 99  # 压缩质量选择范围[50-99]
			# file_path = "save.jpg"
			# stConvertParam.enImageType = MV_Image_Jpeg
			# bmpsize = nPayloadSize

			file_path = "test.bmp"
			stConvertParam.enImageType = MV_Image_Bmp

			bmpsize = stDeviceList.nWidth * stDeviceList.nHeight * 3 + 54

			stConvertParam.nBufferSize = bmpsize
			bmp_buf = (c_ubyte * bmpsize)()
			stConvertParam.pImageBuffer = bmp_buf

			ret = self.cam.MV_CC_SaveImageEx2(stConvertParam)
			if ret != 0:
				print("save file executed failed0:! ret[0x%x]" % ret)
				del data_buf
				sys.exit()
			# print(stop - start)
			# file_open = open(file_path.encode('ascii'), 'wb+')
			try:
				img_buff = (c_ubyte * stConvertParam.nImageLen)()
				cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pImageBuffer, stConvertParam.nImageLen)
				# file_open.write(img_buff, )
				print(img_buff)
				image = np.asarray(bytearray(img_buff), dtype="uint8")
				image = cv2.imdecode(image, cv2.IMREAD_COLOR)
			# temp = np.array(img_buff, dtype=np.uint8)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
			# print(temp.shape)
			# temp = temp.reshape((3072,2048, 3))  # 根据自己分辨率进行转化
			# print(temp)

			# temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB) # 这一步获取到的颜色不对，因为默认是BRG，要转化成RGB，颜色才正常
			# cv2.namedWindow("result", 0)
			# cv2.imshow("result", image)
			# cv2.waitKey(0)
			except Exception as e:
				raise e

		else:
			print("get one frame fail, ret[0x%x]" % ret)

		return image

	def release(self):
		del self

	def __del__(self):
		# ch:停止取流 | en:Stop grab image
		ret = self.cam.MV_CC_StopGrabbing()
		if ret != 0:
			print("stop grabbing fail! ret[0x%x]" % ret)
			del self.data_buf

		# ch:关闭设备 | Close device
		ret = self.cam.MV_CC_CloseDevice()
		if ret != 0:
			print("close deivce fail! ret[0x%x]" % ret)
			del self.data_buf

		# ch:销毁句柄 | Destroy handle
		ret = self.cam.MV_CC_DestroyHandle()
		if ret != 0:
			print("destroy handle fail! ret[0x%x]" % ret)

		del self.data_buf

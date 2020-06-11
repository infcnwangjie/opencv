# -*- coding: utf-8 -*-
import cv2

from app.config import SDK_OPEN
from app.core.exceptions.allexception import SdkException


# from app.core.video.sdk import SdkHandle
from app.core.video.sdk import SdkHandle


class ImageProvider(object):
	'''部署到工控机上使用sdk获取图像，如果测试阶段使用opencv视频库来获取图像'''

	def __init__(self, videofile=None, ifsdk=SDK_OPEN):
		self.videofile = videofile
		self.ifsdk = ifsdk
		if videofile:
			self.IMG_HANDLE = cv2.VideoCapture(videofile)
		elif self.ifsdk:
			self.IMG_HANDLE = SdkHandle() #开启海康sdk调用
			# pass
		else:
			#开启本机摄像头
			self.IMG_HANDLE = cv2.VideoCapture(0)

	def read(self):
		'''从sdk或者opencv获取一帧图像'''
		try:
			imageinfo = self.IMG_HANDLE.read()
		except Exception as e:
			raise e
		else:
			if isinstance(imageinfo, tuple):
				return imageinfo[1]
			else:
				return imageinfo

	def __del__(self):
		'''释放sdk句柄，或者opencv句柄'''
		try:
			if hasattr(self,'IMG_HANDLE') and self.IMG_HANDLE is not None:
				self.IMG_HANDLE.release()
		except SdkException as e:
			raise e

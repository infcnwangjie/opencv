# -*- coding: utf-8 -*-
import cv2

from app.core.exceptions.allexception import SdkException

# from app.core.video.sdk import SdkHandle
from app.core.video.sdk import SdkHandle


class ImageProvider(object):
	'''部署到工控机上使用sdk获取图像，如果测试阶段使用opencv视频库来获取图像'''

	def __init__(self, videofile=None):
		self.videofile = videofile
		if videofile:
			self.IMG_HANDLE = cv2.VideoCapture(videofile)
		else :
			self.IMG_HANDLE = SdkHandle()  # 开启海康sdk调用

	# 读取视频播放速率
	def get_play_speed(self):
		if self.ifsdk:
			return -1
		return self.IMG_HANDLE.get(cv2.CAP_PROP_FPS)



	# 设置播放速率
	def set_play_speed(self, value=None):

		if self.ifsdk:
			# 实时视频数据无法加速
			return

		current_fps = self.get_play_speed()
		if current_fps == -1: return

		if value > 0:
			self.IMG_HANDLE.set(cv2.CAP_PROP_FPS, current_fps + 2 if value is None else current_fps+value)
		else:
			self.IMG_HANDLE.set(cv2.CAP_PROP_FPS, current_fps - 2 if value is None else current_fps-value)


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
			if hasattr(self, 'IMG_HANDLE') and self.IMG_HANDLE is not None:
				self.IMG_HANDLE.release()
		except SdkException as e:
			raise e

import cv2

from app.core.exceptions.allexception import SdkException


# from app.core.video.sdk import SdkHandle


class ImageProvider(object):
	'''部署到工控机上使用sdk获取图像，如果测试阶段使用opencv视频库来获取图像'''

	def __init__(self, videofile=None, ifsdk=True):
		self.videofile = videofile
		self.ifsdk = ifsdk
		if videofile and ifsdk == False:
			self.IMG_HANDLE = cv2.VideoCapture(videofile)
		else:
			# sdk还没开始调研
			pass
			# self.IMG_HANDLE = SdkHandle()

	def read(self):
		'''从sdk或者opencv获取一帧图像'''
		try:
			imageinfo = self.IMG_HANDLE.read()
		except:
			raise SdkException("sdk还没开始调研")
		else:
			if isinstance(imageinfo, tuple):
				return imageinfo[1]
			else:
				raise SdkException("sdk返回什么类型仍未知")

	def __del__(self):
		'''释放sdk句柄，或者opencv句柄'''
		try:
			self.IMG_HANDLE.release()
		except SdkException as e:
			raise e

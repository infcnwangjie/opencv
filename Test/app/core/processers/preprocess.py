# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.processers import SmallWords
from app.core.support.shapedetect import ShapeDetector


class AbstractDetector(metaclass=SmallWords):
	'''
	AbstractDetector
	：used freequently
	'''

	def __init__(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img
		self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		self.img_after_modify=None


	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	# 图像锐化操作
	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst

	# 对灰度图像做数据插值运算
	def interpolation_binary_data(self, binary_image):
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	def enhanceimg(self):
		rows, cols = self.shape
		self.hsv[:, 0:int(0.5 * cols), 2] += 3
		self.hsv[:, int(0.5 * cols + 1):cols, 2] += 10
		self.hsv = self.hsv

	def color_similar_ratio(self, image1, image2):
		'''两张图片的相似度'''
		if image1 is None or image2 is None:
			return 0
		img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		# cv2.imshow("hist1",hist1)
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
		return degree

	def red_contours(self):
		'''返回红色轮廓'''
		red_low, red_high = [120, 50, 50], [180, 255, 255]
		red_min, red_max = np.array(red_low), np.array(red_high)
		# 去除颜色范围外的其余颜色
		red_mask = cv2.inRange(self.hsv, red_min, red_max)
		ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
		red_binary = cv2.medianBlur(red_binary, 3)
		red_contours, _hierarchy = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return red_binary, red_contours

	def yellow_contours(self):
		'''
		返回黄色轮廓
		:return:
		'''
		yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]

		yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)
		# 去除颜色范围外的其余颜色
		yellow_mask = cv2.inRange(self.hsv, yellow_min, yellow_max)

		yellow_ret, yellow_binary = cv2.threshold(yellow_mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		yellow_binary = cv2.medianBlur(yellow_binary, 3)

		yellow_contours, _hierarchy = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return yellow_binary, yellow_contours

	def green_contours(self):
		'''
		返回黄色轮廓
		:return:
		'''
		green_low, green_high = [35, 43, 46], [77, 255, 255]
		# green_low, green_high = [17, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		green_mask = cv2.inRange(self.hsv, green_min, green_max)
		green_ret, green_binary = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)
		green_binary = cv2.medianBlur(green_binary, 3)
		green_contours, _hierarchy = cv2.findContours(green_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return green_binary, green_contours

	def modify_x(self,img):
		'''
		行车中有两条平行的线，这两个平行的线的x轴可以用来更好的标注袋子的X轴坐标
		:return:
		'''

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		edges = cv2.Canny(gray, 50, 150, apertureSize=3)

		cv2.imshow("canny",edges)

		minLineLength = 200
		maxLineGap = 15
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

		targetimg = img.copy()

		for x1, y1, x2, y2 in lines[0]:
			cv2.line(targetimg, (x1, y1), (x2, y2), (0, 255, 0), 2)



		cv2.imshow('targetimg', targetimg)
		cv2.waitKey(0)


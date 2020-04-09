# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 发现
from app.core.target_detect.histcalcute import calculate
from app.core.target_detect.shapedetect import ShapeDetector


class Preprocess(object):
	'''
	预处理操作都在这里
	'''

	def __init__(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img
		self.shapedetector = ShapeDetector()

	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	# 直方图正规化
	def enhance_histrg(self, img):
		Imin, Imax = cv2.minMaxLoc(img)[:2]
		# 使用numpy计算
		# Imax = np.max(img)
		# Imin = np.min(img)
		Omin, Omax = 0, 255
		# 计算a和b的值
		a = float(Omax - Omin) / (Imax - Imin)
		b = Omin - a * Imin
		out = a * img + b
		out = out.astype(np.uint8)
		return out

	# 图像锐化操作
	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst

	# 过滤轮廓
	def filter_landmark_contours(self, c):
		# if not self.shapedetector.detect(c,4):
		# 	return False
		x, y, w, h = cv2.boundingRect(c)
		if w < 20 or h < 20:
			return False

		if not 100 < cv2.contourArea(c) < 80000:
			return False

		# targetimg = self.img[y:y + h, x:x + w]
		# for template_img_path in TEMPLATES_PATH:
		# 	try:
		# 		template_img = cv2.imread(template_img_path)
		# 		match_result = calculate(template_img, targetimg)
		# 	except Exception as e:
		# 		continue
		# 	else:
		# 		print(match_result)
		# 		if match_result > 0.5:
		# 			return True
		#
		# for neg_template_path in NEG_TEMPLATES_PATH:
		# 	neg_template_img = cv2.imread(neg_template_path)
		# 	try:
		# 		neg_match_result = calculate(neg_template_img, targetimg)
		# 	except:
		# 		continue
		# 	else:
		# 		if neg_match_result > 0.45:
		# 			return False

		return True

	# 普通二值化操作
	def find_landmark_contours(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		# self.img = self.enhance_histrg(gray)
		# cv2.namedWindow("gray", 0)
		# cv2.imshow("gray", gray)
		# gray = cv2.equalizeHist(gray)
		# gray=self.enhance_histrg(gray)
		rows, cols = gray.shape
		colorlow = [120, 50, 50]
		colorhigh = [180, 255, 255]
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		binary = cv2.medianBlur(binary, 3)

		# cv2.imshow("first",binary)
		contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		all_contours = list(filter(lambda c: self.filter_landmark_contours(c), contours))
		return all_contours, binary

	# 对灰度图像做数据插值运算
	def interpolation_binary_data(self, binary_image):
		# rows, cols = binary_image.shape
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	# 找到地标的轮廓
	def find_contours_bylandmark_colorrange(self):
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colorlow = (61, 83, 31)
		colorhigh = (81, 255, 250)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		# mask = cv2.erode(mask, None, iterations=3)

		ret, binary = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)
		binary = self.sharper(binary)
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		contours = list(filter(lambda c: self.filter_landmark_contours(c), contours))

		allzero = np.zeros_like(binary)
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			allzero[y:y + h, x:x + w] = binary[y:y + h, x:x + w]
		return contours, allzero

	# 找到袋子轮廓
	def find_contours_bybagcolorrange(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(gray, 70, 150, cv2.THRESH_BINARY)  # 灰度阈值
		# 对binary去噪，腐蚀与膨胀
		binary = cv2.erode(binary, None, iterations=3)
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
		return contours[0:20], binary

	# 找到蓝色小车轮廓
	def find_contours_bybluecarcolorrange(self):
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array([100, 43, 46]), np.array([124, 255, 255])
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
		return contours[0:10], binary

	# 获取已处理过的二值化图像
	@property
	def processedlandmarkimg(self):
		# img1 = self.img.copy()

		# contours, binary = self.find_contours_bylandmark_colorrange()
		contours, binary = self.find_landmark_contours()
		# print("contours num is {}".format(len(contours)))

		return binary, contours

	@property
	def processed_bag(self):
		colorlow = [120, 50, 50]
		colorhigh = [180, 255, 255]
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		binary = cv2.medianBlur(binary, 3)
		contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return binary, contours

	@property
	def processed_laster(self):
		def filter_laster_contour(c):
			rows,cols=self.shape
			x, y, w, h = cv2.boundingRect(c)
			if w<20 or h<20:
				return False
			if 0.32 * cols < x+0.5*w < 0.72 * cols:
				return True
			else:
				return False
		colorlow = [35, 43, 46]
		colorhigh = [77, 255, 255]
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		# binary = cv2.medianBlur(binary, 3)
		# cv2.namedWindow("hockbinaray",0)
		# cv2.imshow("hockbinaray",binary)
		contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		goodcontours = list(filter(filter_laster_contour, contours))

		return binary, goodcontours

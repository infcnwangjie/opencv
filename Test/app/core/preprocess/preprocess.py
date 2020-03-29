import cv2
import numpy as np

# 发现
from app.config import FIRST_TEMPLATE_PATH, THIRD_TEMPLATE_PATH, FIRST_NEG_TEMPLATE_PATH, \
	SECOND_NEG_TEMPLATE_PATH, DETECT_BY_MULTIPLEAREA, DEBUG
from app.core.target_detect.histcalcute import calculate
from app.core.target_detect.shapedetect import ShapeDetector


class Preprocess(object):
	def __init__(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img
		self.shapedetector = ShapeDetector()

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
	def filter_contours(self, c):
		# if not self.shapedetector.detect(c,4):
		# 	return False
		x, y, w, h = cv2.boundingRect(c)
		if w < 20 or h < 20 or w > 100 or h > 100:
			return False

		if not 900 < cv2.contourArea(c) < 20000:
			return False

		# if not isRectange(c):
		# 	return False

		first_template = cv2.imread(FIRST_TEMPLATE_PATH)
		second_template = cv2.imread(FIRST_TEMPLATE_PATH)
		third_template = cv2.imread(THIRD_TEMPLATE_PATH)

		targetimg = self.img[y:y + h, x:x + w]
		first_match_result = calculate(first_template, targetimg)
		second_match_result = calculate(second_template, targetimg)
		third_match_result = calculate(third_template, targetimg)
		if first_match_result > 0.45 or second_match_result > 0.45 or third_match_result > 0.45:
			return True

		neg_template1 = cv2.imread(FIRST_NEG_TEMPLATE_PATH)
		neg_template2 = cv2.imread(SECOND_NEG_TEMPLATE_PATH)
		neg_firstmatch_result = calculate(neg_template1, targetimg)
		neg_secondmatch_result = calculate(neg_template2, targetimg)
		if neg_firstmatch_result > 0.4 or neg_secondmatch_result > 0.4:
			return False
		return True

	# 普通二值化操作
	def find_contours_byeasyway(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		# cv2.namedWindow("gray", 0)
		# cv2.imshow("gray", gray)
		# gray = cv2.equalizeHist(gray)
		# gray=self.enhance_histrg(gray)
		rows, cols = gray.shape
		# 行车环境，左侧部分光线较明亮
		left_gray = gray[:, 0:round(cols / 2 - 1)]
		# 110 能检测出06 07 ；40 能检测出05
		left_ret, left_binary = cv2.threshold(left_gray, 110, 250, cv2.THRESH_BINARY)  # 灰度阈值
		# 40
		left_ret_support, left_binary_support = cv2.threshold(left_gray, 40, 250, cv2.THRESH_BINARY)  # 灰度阈值

		# 行车环境，右侧部分光线较暗
		right_gray = gray[:, round(cols / 2):]
		# 60
		right_ret, right_binary = cv2.threshold(right_gray, 60, 250, cv2.THRESH_BINARY)  # 灰度阈值

		binary = np.zeros_like(gray)
		binary[:, 0:round(cols / 2 - 1)] = left_binary
		binary[:, round(cols / 2):] = right_binary

		# cv2.namedWindow("binary", 0)
		# cv2.imshow("binary", binary)

		# binary = self.sharper(binary)#图像锐化
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		support_left_contours, _drop = cv2.findContours(left_binary_support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = list(filter(lambda c: self.filter_contours(c), contours + support_left_contours))
		support_left_contours = list(filter(lambda c: self.filter_contours(c), support_left_contours))

		allzero = np.zeros_like(binary)
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			allzero[y:y + h, x:x + w] = binary[y:y + h, x:x + w]

		for support_left_contour in support_left_contours:
			x, y, w, h = cv2.boundingRect(support_left_contour)
			allzero[y:y + h, x:x + w] = left_binary_support[y:y + h, x:x + w]

		all_contours = contours + support_left_contours

		return all_contours, allzero

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

		contours = list(filter(lambda c: self.filter_contours(c), contours))

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
		img1 = self.img.copy()
		# contours, binary = self.find_contours_bylandmark_colorrange()
		contours, binary = self.find_contours_byeasyway()
		return binary, contours

	@property
	def processed_bag(self):
		colorlow=[120, 50, 50]
		colorhigh=[180, 255, 255]
		hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		colormin, colormax = np.array(colorlow), np.array(colorhigh)
		# 去除颜色范围外的其余颜色
		mask = cv2.inRange(hsv, colormin, colormax)
		ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
		# 去噪
		binary = cv2.medianBlur(binary, 3)
		contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return binary,contours

	@property
	def processed_laster(self):
		colorlow = [26, 43, 46]
		colorhigh = [34, 255, 255]
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
		return binary, contours




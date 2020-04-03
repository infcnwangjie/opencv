import cv2
import numpy as np

# 发现
from app.config import DETECT_BY_MULTIPLEAREA, TEMPLATES_PATH, NEG_TEMPLATES_PATH
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
	def filter_landmark_contours(self, c):
		# if not self.shapedetector.detect(c,4):
		# 	return False
		x, y, w, h = cv2.boundingRect(c)
		if w < 20 or h < 20 or w > 100 or h > 100:
			return False

		if not 300 < cv2.contourArea(c) < 20000:
			return False

		targetimg = self.img[y:y + h, x:x + w]
		for template_img_path in TEMPLATES_PATH:
			try:
				template_img = cv2.imread(template_img_path)
				match_result = calculate(template_img, targetimg)
			except Exception as e:
				continue
			else:
				print(match_result)
				if match_result > 0.5:
					return True

		for neg_template_path in NEG_TEMPLATES_PATH:
			neg_template_img = cv2.imread(neg_template_path)
			try:
				neg_match_result = calculate(neg_template_img, targetimg)
			except:
				continue
			else:
				if neg_match_result > 0.45:
					return False

		return True


	# 普通二值化操作
	def find_landmark_contours(self):
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
		left_ret_support, left_binary_support_40 = cv2.threshold(left_gray, 40, 250, cv2.THRESH_BINARY)  # 灰度阈值
		# 60
		left_ret_support_60, left_binary_support60 = cv2.threshold(left_gray, 60, 250, cv2.THRESH_BINARY)  # 灰度阈值
		support_left_contours_40, _drop = cv2.findContours(left_binary_support_40, cv2.RETR_EXTERNAL,
		                                                   cv2.CHAIN_APPROX_SIMPLE)

		support_left_contours_60, _drop = cv2.findContours(left_binary_support60, cv2.RETR_EXTERNAL,
		                                                   cv2.CHAIN_APPROX_SIMPLE)

		#
		# left_binary_firststep = cv2.bitwise_or(left_binary, left_binary_support_40)
		# left_binary_secondstep = cv2.bitwise_or(left_binary, left_binary_support60)

		# cv2.imshow("left_binary_support60",left_binary_support60)

		# 行车环境，右侧部分光线较暗
		right_gray = gray[:, round(cols / 2):]
		# 60

		right_ret, right_binary = cv2.threshold(right_gray, 60, 250, cv2.THRESH_BINARY)  # 灰度阈值
		right_ret, right_binary_40 = cv2.threshold(right_gray, 40, 250, cv2.THRESH_BINARY)  # 灰度阈值
		right_contours_40, _drop = cv2.findContours(right_binary_40, cv2.RETR_EXTERNAL,
		                                            cv2.CHAIN_APPROX_SIMPLE)

		binary = np.zeros_like(gray)
		binary[:, 0:round(cols / 2 - 1)] = left_binary
		binary[:, round(cols / 2):] = right_binary

		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# cv2.namedWindow("first_contours", 0)
		# cv2.imshow("first_contours", self.img)

		allzero = np.zeros_like(binary)
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			allzero[y:y + h, x:x + w] = binary[y:y + h, x:x + w]

		for support_left_contour in support_left_contours_40:
			x, y, w, h = cv2.boundingRect(support_left_contour)
			allzero[y:y + h, x:x + w] = left_binary_support_40[y:y + h, x:x + w]

		for support_left_contour in support_left_contours_60:
			x, y, w, h = cv2.boundingRect(support_left_contour)
			allzero[y:y + h, x:x + w] = left_binary_support60[y:y + h, x:x + w]

		for support_right_contour in right_contours_40:
			x, y, w, h = cv2.boundingRect(support_right_contour)
			allzero[y:y + h, x:x + w] = right_binary_40[y:y + h, x:x + w]

		all_contours = contours + support_left_contours_40 + support_left_contours_60
		all_contours = list(filter(lambda c: self.filter_landmark_contours(c), all_contours))

		cv2.namedWindow("final_binary", 0)
		cv2.imshow("final_binary", allzero)
		#
		# cv2.namedWindow("temp_contours", 0)
		#
		# cv2.drawContours(self.img, all_contours, -1, (0, 255, 255), 5)
		# cv2.imshow("temp_contours", self.img)

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
		img1 = self.img.copy()
		# contours, binary = self.find_contours_bylandmark_colorrange()
		contours, binary = self.find_landmark_contours()
		print("contours num is {}".format(len(contours)))

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

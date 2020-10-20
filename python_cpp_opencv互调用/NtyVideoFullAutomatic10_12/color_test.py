import cv2
import numpy as np
# ------------------------------------------------
# 名称：BaseDetector
# 功能：目标检测基类，提供一些公共的方法
# 状态：在用
# 作者：王杰  注释后补
# ------------------------------------------------
from app.core.video.mvs.MvCameraSuppl_class import MvSuply, IMG_HEIGHT, IMG_WIDTH


class BaseDetector(object):
	orb_detector = None  #

	# ------------------------------------------------
	# 名称：logger
	# 功能：封装日志功能
	# 状态：在用
	# 参数： [msg]   ---日志消息
	#        [lever] ---日志级别
	# 返回：None ---
	# 作者：王杰  2020-5-xx
	# ------------------------------------------------
	def logger(self, msg: str, lever='info'):
		from app.log.logtool import logger
		logger(msg, lever)

	# def filter_matches(kp1, kp2, matches, ratio=0.75):
	# 	mkp1, mkp2 = [], []
	# 	for m in matches:
	# 		if  m.distance < m.distance * ratio:
	# 			m = m[0]
	# 			mkp1.append(kp1[m.queryIdx])
	# 			mkp2.append(kp2[m.trainIdx])
	# 	p1 = [kp.pt for kp in mkp1]
	# 	p2 = [kp.pt for kp in mkp2]
	# 	# kp_pairs = zip(mkp1, mkp2)
	# 	return p1, p2

	# 获取图像行、列总数
	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	# ------------------------------------------------
	# 名称：sharper
	# 功能：凸显边缘信息
	# 状态：在用
	# 参数： [array]
	# 返回： array --- 输出图像
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst

	# ------------------------------------------------
	# 名称：interpolation_binary_data
	# 功能：插值
	# 状态：在用
	# 参数： [array]
	# 返回： array
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def interpolation_binary_data(self, binary_image):
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	def color_similar_ratio(self, image1, image2):
		if image1 is None or image2 is None:
			return 0
		try:
			degree = float(MvSuply.SAME_RATE(image1, image2))
		except Exception as e:
			degree = 0
		return degree

	# ------------------------------------------------
	# 名称：red_contours
	# 功能：获取红色轮廓
	# 状态：在用
	# 参数： [array]
	#        [int]
	#        [int]
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def red_contours(self, img, middle_start=180, middle_end=500):
		# red_low, red_high = [120, 50, 50], [180, 255, 255]
		red_low, red_high = [156, 43, 46], [180, 255, 255]
		red_min, red_max = np.array(red_low), np.array(red_high)
		# 去除颜色范围外的其余颜色
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		red_mask = cv2.inRange(hsv, red_min, red_max)
		ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
		middle_open_mask = np.zeros_like(red_binary)
		middle_open_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255
		red_binary = cv2.bitwise_and(red_binary, red_binary, mask=middle_open_mask)
		red_binary = cv2.medianBlur(red_binary, 3)
		# cv2.imshow("red_b",red_binary)
		red_contours, _hierarchy = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return red_binary, red_contours

	# ------------------------------------------------
	# 名称：red_contours
	# 功能：获取黄色轮廓
	# 状态：在用
	# 参数： [array]   ---输入图像
	# 要求： img是RGB格式图片
	# 返回： 数组     ---轮廓数组
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def yellow_contours(self, img):
		yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]

		yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)
		# 去除颜色范围外的其余颜色

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

		yellow_ret, yellow_binary = cv2.threshold(yellow_mask, 100, 255, cv2.THRESH_BINARY)

		yellow_contours, _hierarchy = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return yellow_binary, yellow_contours

	# ------------------------------------------------
	# 名称：green_contours
	# 功能：获取绿色轮廓
	# 状态：在用
	# 参数：  [array]
	#         [int]
	#         [int]
	# 要求： img是RGB格式图片
	# 作者：王杰  编写 2020-4-xx  修改 2020-6-xx
	# ------------------------------------------------
	def green_contours(self, img, middle_start=100, middle_end=450):
		rows, cols, channels = img.shape
		# 如果尺寸已经调整，就无须调整
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# cv2.imshow("hsv",hsv)
		# green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		green_mask = cv2.inRange(hsv, green_min, green_max)

		green_ret, foreground = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)

		green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return foreground, green_contours

	def point_in_contour(self, x, y, c):
		c_x, c_y, c_w, c_h = cv2.boundingRect(c)

		top1_x, top1_y, top2_x, top2_y = c_x, c_y, c_x + c_w, c_y
		top3_x, top3_y, top4_x, top4_y = c_x, c_y + c_h, c_x + c_w, c_y + c_h

		if top1_x < x < top2_x and top1_y < y < top3_y:
			return True
		else:
			return False
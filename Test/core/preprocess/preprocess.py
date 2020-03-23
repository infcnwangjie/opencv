import cv2
import numpy as np
import matplotlib.pyplot as plt


# 发现
class Preprocess(object):
	def __init__(self, img):
		if isinstance(img, str):
			self.img = cv2.imread(img)
		else:
			self.img = img

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
		x, y, w, h = cv2.boundingRect(c)
		if w < 20 or h < 20 or w > 100 or h > 100:
			return False
		if not 800 < cv2.contourArea(c) < 20000:
			return False
		return True

	# 普通二值化操作
	def find_contours_byeasyway(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		# gray = cv2.equalizeHist(gray)
		# gray=self.enhance_histrg(gray)
		rows, cols = gray.shape
		# 行车环境，左侧部分光线较明亮
		left_gray = gray[:, 0:round(cols / 2 - 1)]
		# 110 能检测出06 07 ；40 能检测出05
		left_ret, left_binary = cv2.threshold(left_gray, 110, 250, cv2.THRESH_BINARY)  # 灰度阈值
		left_ret_support, left_binary_support = cv2.threshold(left_gray, 40, 250, cv2.THRESH_BINARY)  # 灰度阈值

		# 行车环境，右侧部分光线较暗
		right_gray = gray[:, round(cols / 2):]
		# 60
		right_ret, right_binary = cv2.threshold(right_gray, 60, 250, cv2.THRESH_BINARY)  # 灰度阈值

		binary = np.zeros_like(gray)
		binary[:, 0:round(cols / 2 - 1)] = left_binary
		binary[:, round(cols / 2):] = right_binary

		# binary = self.sharper(binary)#图像锐化
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		support_left_contours, _drop = cv2.findContours(left_binary_support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = list(filter(lambda c: self.filter_contours(c), contours + support_left_contours))
		support_left_contours = list(filter(lambda c: self.filter_contours(c), support_left_contours))
		# contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

		allzero = np.zeros_like(binary)
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			allzero[y:y + h, x:x + w] = binary[y:y + h, x:x + w]

		for support_left_contour in support_left_contours:
			x, y, w, h = cv2.boundingRect(support_left_contour)
			allzero[y:y + h, x:x + w] = left_binary_support[y:y + h, x:x + w]

		all_contours = contours + support_left_contours
		cv2.namedWindow("easy_binary", 0)
		cv2.imshow("easy_binary", allzero)
		return all_contours, allzero

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
		cv2.namedWindow("landmark_binary", 0)
		cv2.imshow("landmark_binary", allzero)
		return contours, allzero

	# 找到袋子轮廓
	def find_contours_bybagcolorrange(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		ret, binary = cv2.threshold(gray, 70, 150, cv2.THRESH_BINARY)  # 灰度阈值
		# 对binary去噪，腐蚀与膨胀
		binary = cv2.erode(binary, None, iterations=3)
		# cv2.namedWindow("bag_detect", 0)
		# cv2.imshow("bag_detect", binary)
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
		# binary.where()
		cv2.namedWindow("car_detect", 0)
		cv2.imshow("car_detect", binary)
		contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
		return contours[0:10], binary

	# 获取已处理过的二值化图像
	@property
	def processedimg(self):
		img1 = self.img.copy()
		landmark_contours, landmark_binary = self.find_contours_bylandmark_colorrange()
		# result1 = cv2.drawContours(img1, landmark_contours, -1,
		#                            (0, 255, 0), 3)
		# bag_contours, bag_binary = self.find_contours_bybagcolorrange()
		# result2 = cv2.drawContours(result1, bag_contours, -1,
		#                            (0, 0, 255), 3)
		# bluecar_contours, bluecar_binary = self.find_contours_bybluecarcolorrange()
		# result3 = cv2.drawContours(result2, bluecar_contours, -1,
		#                            (255, 0, 0), 3)

		easy_contours, easy_binary = self.find_contours_byeasyway()
		result = cv2.drawContours(img1, easy_contours, -1,
		                          (0, 0, 255), 3)
		# finaly = np.zeros_like(landmark_binary)
		# 发现hsv检测出来的方式，轮廓大致保存完整，不完整的部分，让普通二进制的方式补充即可
		# 1：该方法找到地标的位置，从灰度图像上拷贝合适的像素过来，相对来说可以
		# for landmarkcontour in landmark_contours:
		#     if cv2.contourArea(landmarkcontour) > 900:
		#         x, y, w, h = cv2.boundingRect(landmarkcontour)
		#         finaly[y + 1:y + h, x + 1:x + w] = easy_binary[y + 1:y + h, x + 1:x + w]
		#         cv2.rectangle(finaly, (x - 8, y - 8), (x + w + 8, y + h + 8), color=255, thickness=3)
		# 2：除了地标位置的像素保留不动，其余的像素都是地标二进制的

		cv2.namedWindow("easycontours", 0)
		cv2.imshow("easycontours", result)

		#
		# plt.figure()
		# plt.subplot(1, 3, 1)
		# plt.imshow(landmark_binary)
		# plt.subplot(1, 3, 2)
		# plt.imshow(easy_binary)
		# plt.subplot(1, 3, 3)
		# plt.imshow(result)
		# plt.show()

		return easy_binary


if __name__ == '__main__':
	process = Preprocess(img="bag8.bmp")
	# img = process.processedimg
	img = process.processedimg

	# cv2.namedWindow("finaly", 0)
	# cv2.imshow("finaly", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

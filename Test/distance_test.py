import math
import time

import cv2
import numpy

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.processers.preprocess import BagDetector, LasterDetector, HockDetector
from app.core.processers.preprocess import LandMarkDetecotr, WITH_TRANSPORT


# TODO 计算物体距离摄像头的距离
def compute_height(W=110, F=6, P=10, H=9500):
	'''
	用于求解袋子高度
	@:param W: 实际袋子条宽度
	@:param F:焦距                   已知项，摄像头焦距 F  6mm
	@:param P:图像像素宽度
	@:param H:摄像头距离地面的高度   已知项，摄像头高度 H　８米
	:return: D=H-W*F/P
	'''
	if P < 0:
		raise Exception("图像像素宽度计算有误")
	try:
		d = W * F / P
	except:
		raise Exception("计算有误")
	return d


def compute_x(L=10, H=8000, F=6, W=100, P=10, D=0):
	'''
	# TODO 计算物体X轴坐标值
	# X*X=L*L -(H-FW/P)*(H-FW/P)
	:param image:
	:return:
	'''
	try:
		a = math.pow(L, 2)
		b = D if D != 0 or D is not None else math.pow(H - F * W / P, 2)
		x = math.sqrt(abs(a - b))
	except Exception as e:
		raise e
	return x


def test_one_image():
	a = LandMarkDetecotr()
	# 	# cap = cv2.VideoCapture("C:/NTY_IMG_PROCESS/VIDEO/Video_20200519095438779.avi")
	# cap.set(cv2.CAP_PROP_POS_FRAMES, 243)
	# ret, frame = cap.read()
	# frame=cv2.resize(frame,(IMG_HEIGHT,IMG_WIDTH))
	# cv2.namedWindow("src")
	# cv2.imshow("src",frame)
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152805570.bmp")
	image = cv2.imread("D:/PIC/Image_20200602124234979.bmp")
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152825743.bmp")
	# image=cv2.imread("c:/work/nty/hangche/Image_20200522154907736.bmp")
	image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

	dest, success = a.position_landmark(image)
	dest_copy = dest.copy()
	bags, foreground = BagDetector().location_bags(dest, dest_copy, middle_start=100, middle_end=500)

	laster, lasterforeground = LasterDetector().location_laster(dest, dest_copy)
	for bag in bags:
		D = compute_height(W=110, F=6, P=bag.width, H=9577)
		print("-" * 100)
		print("高度为：{}".format(D))
		print(bag)
		print("计算坐标：{}".format(compute_x(bag.x * 10, H=9577, F=6, W=110, P=bag.width, D=0)))
	a.draw_grid_lines(dest)

	# cap.release()
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)

	cv2.namedWindow("laster")
	cv2.imshow("laster", lasterforeground)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


'''
	定位袋子
	#定位袋子要求，将每个袋子划定区域边界，并且对边界编号，每次实时更新都能满足不变性，除非袋子已经拿走
	决定抓取哪个袋子
	#决定抓取哪个袋子，是计算钩子与袋子区域质心的距离，决定优先抓哪个袋子
	实时计算钩子与袋子的距离
	if 距离达到阈值要求：
		move
	else:
		落钩
'''


def avi_without_hock():
	# Video_20200605142854079.avi
	import cv2

	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200602132439483.avi")  # 打开相机
	# cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200602132439483.avi")  # 打开相机
	a = LandMarkDetecotr()
	b = BagDetector()
	c = LasterDetector()
	d = HockDetector()
	# 背景差分法

	middle_start = 120
	middle_end = 570
	while (True):
		ret, frame = cap.read()  # 捕获一帧图像
		time.sleep(1 / 13)
		frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if ret:
			dest, success = a.position_landmark(frame)

			# cv2.imshow("fgmask", fgmask)
			dest_copy = dest.copy()
			# zero = numpy.zeros_like(gray)
			# zero[0:IMG_HEIGHT, middle_start:middle_end] = 255

			if success:
				print("landmark 定位:{}".format(success))

				bags, foreground = b.location_bags(dest, dest_copy, middle_start=middle_start, middle_end=middle_end)
				#
				# laster, lasterforeground = c.location_laster(dest, dest_copy, middle_start=120, middle_end=450)
				hock, hockforeground = d.location_hock(dest, dest_copy, middle_start=middle_start,
				                                       middle_end=middle_end)

				cv2.imshow("hock_foreground", hockforeground)

			cv2.imshow('frame', dest)
			cv2.waitKey(1)
		else:
			break

	cap.release()  # 关闭相机
	cv2.destroyAllWindows()  # 关闭窗


def match_ORB(img2):
	img1 = cv2.imread('D:/NTY_IMG_PROCESS/HOCK_ROI/HOCK_.png', 0)
	# img2 = cv2.imread('./gggg/002.png',0)

	# 使用ORB特征检测器和描述符，计算关键点和描述符
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)

	img3 = cv2.drawMatches(img1=img1, keypoints1=kp1,
	                       img2=img2, keypoints2=kp2,
	                       matches1to2=matches,
	                       outImg=img2, flags=2)
	return img3


def avi_play():
	import cv2

	# cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200520142647832.avi")  # 打开相机
	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200609142146157.avi")  # 打开相机
	a = LandMarkDetecotr()
	b = BagDetector()
	# c = LasterDetector()
	d = HockDetector()
	# 背景差分法
	# fgbg  = cv2.createBackgroundSubtractorMOG2()
	cv2.namedWindow("foreground")
	while (True):
		ret, frame = cap.read()  # 捕获一帧图像
		time.sleep(1 / 13)
		frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
		if ret:
			dest, success = a.position_landmark(frame)
			# fgmask = fgbg.apply(frame)
			# cv2.imshow("fgmask",fgmask)
			dest_copy = dest.copy()
			# match_img=match_ORB(dest_copy)
			# cv2.imshow("match",match_img)

			# positions=numpy.argwhere(dest_copy==0)
			# for index,positionlist in enumerate(positions):
			# 	print(index)
			# 	print(positionlist)
			# print(positions)

			if success:
				print("landmark 定位:{}".format(success))
				bags, foreground = b.location_bags(dest, dest_copy, middle_start=120, middle_end=450)
				_1, foreground = d.location_hock(dest, dest_copy)
				if foreground is not None:
					cv2.imshow("foreground", foreground)

			# laster, lasterforeground = c.location_laster(dest, dest_copy, middle_start=120, middle_end=450)

			cv2.imshow('frame', dest)
			cv2.waitKey(1)
		else:
			break

	cap.release()  # 关闭相机
	cv2.destroyAllWindows()  # 关闭窗


def time_test():
	time_str = time.strftime("%Y%m%d%X", time.localtime()).replace(":", "")
	print(time_str)


def hock_play():
	import cv2

	# cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200520142647832.avi")  # 打开相机
	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200609142146157.avi")  # 打开相机
	a = LandMarkDetecotr()
	b = BagDetector()
	# c = LasterDetector()
	d = HockDetector()
	# 背景差分法
	# fgbg  = cv2.createBackgroundSubtractorMOG2()
	# cv2.namedWindow("foreground")
	while (True):
		ret, frame = cap.read()  # 捕获一帧图像
		time.sleep(1 / 13)
		frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
		if ret:
			# dest, success = a.position_landmark(frame)
			# dest_copy = dest.copy()
			#
			# if success:
			# 	print("landmark 定位:{}".format(success))
			# 	bags, bagf = b.location_bags(dest, dest_copy, middle_start=120, middle_end=500)
			# 	cv2.imshow("bagf",bagf)
			# 	_1, foreground = d.location_hock(dest, dest_copy)
			# 	if foreground is not None:
			# 		cv2.imshow("foreground", foreground)


			bags, bagf = b.location_bags(frame, frame.copy(),success_location=False, middle_start=240, middle_end=500)
			# cv2.imshow("bag",bagf)
			_1, foreground = d.location_hock(frame, frame.copy(),find_landmark=False,middle_start=245,middle_end=490)
			# if foreground is not None:
			# 		cv2.imshow("hock", foreground)
			cv2.imshow('frame', frame)
			cv2.waitKey(1)
		else:
			break

	cap.release()  # 关闭相机
	cv2.destroyAllWindows()  # 关闭窗


#
# def quick_sort(arr):
#     """快速排序"""
#     if len(arr) < 2:
#         return arr
#     # 选取基准，随便选哪个都可以，选中间的便于理解
#     mid = arr[len(arr) // 2]
#     # 定义基准值左右两个数列
#     left, right = [], []
#     # 从原始数组中移除基准值
#     arr.remove(mid)
#     for item in arr:
#         # 大于基准值放右边
#         if item.x >= mid.x:
#             right.append(item)
#         else:
#             # 小于基准值放左边
#             left.append(item)
#     # 使用迭代进行比较
#     return quick_sort(left) + [mid] + quick_sort(right)
#
#
# def test_sort():
# 	b = [11, 99, 33, 69, 77, 88, 55, 11, 33, 36, 39, 66, 44, 22]
# 	c=quick_sort(b)
# 	for num in c:
# 		print(num)


def three_frame_differencing():
	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200602132439483.avi")
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	one_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	two_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	three_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	while cap.isOpened():
		ret, frame = cap.read()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if not ret:
			break
		one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
		abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
		_, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

		abs2 = cv2.absdiff(two_frame, three_frame)
		_, thresh2 = cv2.threshold(abs2, 40, 255, cv2.THRESH_BINARY)

		binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		erode = cv2.erode(binary, kernel)  # 腐蚀
		dilate = cv2.dilate(erode, kernel)  # 膨胀
		dilate = cv2.dilate(dilate, kernel)  # 膨胀

		contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
		                                 method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

		# cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for contour in contours:
		# 	if 100 < cv2.contourArea(contour) < 40000:
		# 		x, y, w, h = cv2.boundingRect(contour)  # 找方框
		# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
		cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
		cv2.imshow("binary", binary)
		# cv2.imshow("dilate", dilate)
		# cv2.imshow("frame", frame)
		if cv2.waitKey(50) & 0xFF == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()


def two_frame_differencing():
	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200602132439483.avi")
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	one_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	two_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	# three_frame = numpy.zeros((height, width), dtype=numpy.uint8)
	while cap.isOpened():
		ret, frame = cap.read()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if not ret:
			break
		one_frame, two_frame = two_frame, frame_gray
		abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
		_, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		erode = cv2.erode(thresh1, kernel)  # 腐蚀
		dilate = cv2.dilate(erode, kernel)  # 膨胀
		# dilate = cv2.dilate(dilate, kernel)  # 膨胀

		contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
		                                 method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

		# cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for contour in contours:
		# 	if 100 < cv2.contourArea(contour) < 40000:
		# 		x, y, w, h = cv2.boundingRect(contour)  # 找方框
		# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
		cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
		# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
		cv2.imshow("binary", dilate)
		# cv2.imshow("dilate", dilate)
		# cv2.imshow("frame", frame)
		if cv2.waitKey(50) & 0xFF == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()


def rp():
	# str_t = r"刘喆 鲁C962EL,18766963345"
	str1_t = r"吕绪文，15065872895,鲁C3U608"
	import re
	match_car_no = re.compile(".*?([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}).*?")

	result=re.match(match_car_no,str1_t)
	if result is not None:
		print(result.group(1))
	else:
		print("fail")


def fangcha():
	a=[1,5,6,7,8,9]

	average_v=numpy.average(a[-4:])
	print(average_v)
	Q2=0
	for item in a[-4:]:
		Q2+=math.pow(item-average_v,2)
	print(math.sqrt(Q2))

	# print(a[-4:])

if __name__ == '__main__':
	# test_one_image()
	# avi_play()
	hock_play()
	# fangcha()
	# rp()
# two_frame_differencing()
# avi_without_hock()
# test_sort()
# time_test()
# numpy_mat()
# three_frame_differencing()

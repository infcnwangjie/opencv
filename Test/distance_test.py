import math
import time

import cv2
import numpy

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.processers.preprocess import BagDetector, LasterDetector
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





def avi_play():
	import cv2

	cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200520142647832.avi")  # 打开相机
	# cap = cv2.VideoCapture("D:/PIC/MV-CA060-10GC (00674709176)/Video_20200602132439483.avi")  # 打开相机
	a = LandMarkDetecotr()
	b = BagDetector()
	c = LasterDetector()
	# 背景差分法
	fgbg  = cv2.createBackgroundSubtractorMOG2()
	while (True):
		ret, frame = cap.read()  # 捕获一帧图像
		time.sleep(1 / 13)
		frame = cv2.resize(frame, (IMG_HEIGHT,IMG_WIDTH))
		if ret:
			dest, success = a.position_landmark(frame)
			fgmask = fgbg.apply(frame)
			cv2.imshow("fgmask",fgmask)
			dest_copy = dest.copy()
			if success:
				print("landmark 定位:{}".format(success))
				bags, foreground = b.location_bags(dest, dest_copy, middle_start=120, middle_end=450)

				laster, lasterforeground = c.location_laster(dest, dest_copy, middle_start=120, middle_end=450)

			cv2.imshow('frame', dest)
			cv2.waitKey(1)
		else:
			break

	cap.release()  # 关闭相机
	cv2.destroyAllWindows()  # 关闭窗


def time_test():
	time_str = time.strftime("%Y%m%d%X", time.localtime()).replace(":", "")
	print(time_str)
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


if __name__ == '__main__':
	# test_one_image()
	avi_play()
	# test_sort()
	# time_test()
	# numpy_mat()

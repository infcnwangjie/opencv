import os

import cv2
import numpy as np

from app.config import TEMPLATES_PATH, NEG_TEMPLATES_PATH, TRAIN_DATA_DIR
from app.core.target_detect.pointlocation import PointLocationService

img1 = cv2.imread('C:/work/imgs/test/2020-04-03-08-13-29test.bmp')
img2 = cv2.imread('C:/work/imgs/test/2020-04-03-09-49-40test.bmp')
img3 = cv2.imread("C:/work/imgs/test/2020-04-03-10-46-41test.bmp")
img4 = cv2.imread("C:/work/imgs/test/2020-04-03-12-21-06test.bmp")
img5 = cv2.imread("C:/work/imgs/test/2020-04-03-13-49-56test.bmp")
img6 = cv2.imread("C:/work/imgs/test/bag6.bmp")

img7 = cv2.imread("C:/work/imgs/test/bag7.bmp")
img8 = cv2.imread("C:/work/imgs/test/2020-04-03-16-05-58test.bmp")
test_img = img6


# find_landmark_contours()

def filter_point(point, img):
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	try:
		rows, cols = img.shape
	except:
		rows, cols, chanels = img.shape

	if abs(point.pt[0] - cols / 2) < 400:
		return False
	if abs(point.pt[0] - cols / 4) < 200:
		return False
	return True


# point cv2.KeyPoint
def drop_unrule_points(img, points):
	scoredict = {}
	try:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except:
		gray = img

	rows, cols = gray.shape
	point_dict = {}
	for a in points:
		scoredict[str(id(a))] = 0
		point_dict[str(id(a))] = a
		for b in points:
			if a == b:
				scoredict[str(id(a))] += 1
			elif abs(a.pt[0] - b.pt[0]) + abs(a.pt[1] - b.pt[1]) < 700:
				continue
			elif abs(b.pt[0] - 0.5 * cols) < 1000:
				continue
			elif abs(a.pt[0] - b.pt[0]) < 50 or abs(a.pt[1] - b.pt[1]) < 50:
				scoredict[str(id(a))] += 1
	good_point_ids = [index for index, score in scoredict.items() if score > 3]

	return [point_dict[id] for id in good_point_ids]


def find_landmark_contours(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# rows, cols = gray.shape
	ret, binary = cv2.threshold(gray, 60, 250, cv2.THRESH_BINARY)  # 灰度阈值
	contours, _drop = cv2.findContours(binary, cv2.RETR_EXTERNAL,
	                                   cv2.CHAIN_APPROX_SIMPLE)

	cv2.namedWindow("final_binary", 0)
	cv2.imshow("final_binary", binary)
	#
	cv2.namedWindow("contour_result", 0)
	#
	cv2.drawContours(img, contours, -1, (0, 255, 255), 5)
	cv2.imshow("contour_result", img)

	return contours, binary


def draw_map(template_img, destimg):
	import numpy as np
	import cv2
	# global img2, img1, img3, img4, img5

	# 读取图片内容
	# img1 = cv2.imread('aa.jpg', 0)
	# img1 = img1
	# img2 = cv2.imread(os.path.join(TRAIN_DATA_DIR, "template1.png"), 0)
	# img2 = img2

	# 使用ORB特征检测器和描述符，计算关键点和描述符
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(template_img, None)
	kp2, des2 = orb.detectAndCompute(destimg, None)
	# kp3, des3 = orb.detectAndCompute(img3, None)
	# kp4, des4 = orb.detectAndCompute(img4, None)
	# kp5, des5 = orb.detectAndCompute(img5, None)
	#
	# kp1=list(filter(lambda point:filter_point(point,img1),kp1))
	# kp2 = list(filter(lambda point: filter_point(point, img2), kp2))

	# 暴力匹配BFMatcher，遍历描述符，确定描述符是否匹配，然后计算匹配距离并排序
	# BFMatcher函数参数：
	# normType：NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2。
	# NORM_L1和NORM_L2是SIFT和SURF描述符的优先选择，NORM_HAMMING和NORM_HAMMING2是用于ORB算法
	bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)
	# matches是DMatch对象，具有以下属性：
	# DMatch.distance - 描述符之间的距离。 越低越好。
	# DMatch.trainIdx - 训练描述符中描述符的索引
	# DMatch.queryIdx - 查询描述符中描述符的索引
	# DMatch.imgIdx - 训练图像的索引。

	# 使用plt将两个图像的匹配结果显示出来
	img3 = cv2.drawMatches(img1=template_img, keypoints1=kp1, img2=destimg, keypoints2=kp2, matches1to2=matches,
	                       outImg=destimg,
	                       flags=2)
	img3 = cv2.resize(img3, (1400, 1000))
	cv2.imshow("dm", img3)

	for point in kp2:
		print(point.pt)

	commonpoints = kp1 and kp2

	# commonpoints=drop_unrule_points(test_img,commonpoints)

	# img3 = cv2.drawKeypoints(img2, commonpoints, None, color=(0, 255, 0),
	#                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

	# for point_obj_a in commonpoints:
	# 	point_a_x, point_a_y = point_obj_a.pt
	# 	for point_obj_b in commonpoints:
	# 		if point_obj_a != point_obj_b:
	# 			pass
	# 		point_b_x, point_b_y = point_obj_b.pt
	#
	# 		if 20 < abs(point_a_x - point_b_x) < 130 and 20 < abs(point_a_y - point_b_y) < 130:
	# 			cv2.line(img3,(int(point_a_x), int(point_a_y)),(int(point_b_x), int(point_b_y)),color=(0, 255, 255),thickness=3)
	points = [[point.pt[0], point.pt[1]] for point in commonpoints if filter_point(point, destimg)]
	print(points)
	point_array = np.float32(points)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS

	ompactness, labels, centers = cv2.kmeans(point_array, 10, None, criteria, 300, flags)
	print(ompactness, labels, centers)

	for center_x, center_y in centers:
		# cv2.circle(img3,(int(center_x),int(center_y)),radius=100,color=(0, 255, 0),thickness=3)
		cv2.rectangle(destimg, (int(center_x) - 50, int(center_y) - 50), (int(center_x) + 50, int(center_y) + 50),
		              color=(0, 255, 0), thickness=3)

	result = cv2.resize(destimg, (1400, 1000))
	cv2.imshow("result", result)


# draw_map(template_img=img2, destimg=img5)

img=cv2.imread("C:/work/imgs/test/2020-04-03-16-15-28test.bmp")
# img=cv2.imread("C:/work/imgs/test/2020-04-03-16-15-28test.bmp")
# img = cv2.imread("C:/work/imgs/test/2020-04-03-16-32-31test.bmp")
with PointLocationService(img=img) as service:
	service.computer_landmarks_location()
# find_landmark_contours(img3)
cv2.namedWindow("landmark", 0)
cv2.imshow("landmark", service.img)
cv2.waitKey(0)
cv2.destroyAllWindows()

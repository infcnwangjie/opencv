# encoding:utf-8
import os



import cv2
import numpy as np

from core.target_detect.pointlocation import PointLocationService, ONLY_LANDMARK, BAG_AND_LANDMARK
from core.target_detect.svmclassify import SvmClass, AnnClass


def rectangle_detect():
	img = cv2.imread('imgs/test/bag1.bmp')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	img=cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
	cv2.namedWindow("window",0)
	cv2.imshow("window",gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def hug_svm_detect_contours():
	win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
	block_size = (16, 16)  # （每个块最大为16 * 16）
	block_stride = (8, 8)  # 单元格尺寸
	cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
	num_bins = 9  # 对于每一个单元格，统计9个方向的梯度直方图。
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
	land_mark_dir = "imgs/land_mark/"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "{dir}/{i}".format(dir=land_mark_dir, i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		x_lands.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.ones(train_lands.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos

	othersdir = "imgs/others/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		x_ohters.append(hog.compute(img, (64, 64)))
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = 3 * np.ones(train_ohers.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "imgs/bags/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		x_bags.append(hog.compute(img, (64, 64)))
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = 2 * np.ones(train_bags.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))
	# labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
	# labels = np.array(labels)

	# labels转换成10行1列的矩阵
	# labels = labels.reshape(10, 1)
	# trainingData转换成10行2列的矩阵
	# trainingData = trainingData.reshape(10, 2)

	# 创建分类器
	svm =SvmClass(trainingData,labels)

	ret = svm.trainData()

	im = cv2.imread('imgs/test/bag1.bmp')
	# 预处理部分
	colorlow = (61, 83, 31)
	colorhigh = (81, 255, 250)
	colormin, colormax = np.array(colorlow), np.array(colorhigh)
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	# 去除颜色范围外的其余颜色
	mask = cv2.inRange(hsv, colormin, colormax)
	ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
	# 去噪
	binary = cv2.medianBlur(binary, 3)

	cv2.namedWindow("binary", 0)
	cv2.imshow("binary", binary)

	contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	moderatesize_countours = []
	moderatesize_rects = []
	boxindex = 0
	for countour in contours:
		countour_rect = cv2.boundingRect(countour)
		rect_x, rect_y, rect_w, rect_h = countour_rect
		if rect_w > 25 and rect_h > 25 and cv2.contourArea(countour) > 300:
			moderatesize_rects.append(countour_rect)
			moderatesize_countours.append(countour)

			testimg = im[rect_y + 1:rect_h + rect_y, rect_x + 1:rect_w + rect_x]
			testimg = cv2.resize(testimg, (50, 50))
			testfeature = hog.compute(testimg, (64, 64))
			testfeature = np.array([testfeature], dtype=np.float32)
			(ret, res) = svm.predictData(testfeature)
			print(ret, str(int(res[0][0])))
			cv2.putText(im, "res:{}".format(str(int(res[0][0]))),
			            (int(rect_x + 0.5 * rect_w + 50), int(rect_y + 0.5 * rect_h + 50)),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

	# for feature, rect in zip(testfeatures, rects):
	# 	pass

	cv2.drawContours(im, moderatesize_countours, -1, (0, 255, 0), 1)

	cv2.namedWindow("im", 0)
	cv2.imshow("im", im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def hug_ann_detect_contours():
	win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
	block_size = (16, 16)  # （每个块最大为16 * 16）
	block_stride = (8, 8)  # 单元格尺寸
	cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
	num_bins = 9  # 对于每一个单元格，统计9个方向的梯度直方图。
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
	land_mark_dir = "imgs/land_mark/"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "{dir}/{i}".format(dir=land_mark_dir, i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		x_lands.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.array(np.ones(train_lands.shape[0], dtype=np.int32),dtype=np.float32 ) # 将训练样本赋值为1给y_pos

	othersdir = "imgs/others/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		x_ohters.append(hog.compute(img, (64, 64)))
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = np.array(3 * np.ones(train_ohers.shape[0], dtype=np.int32) ,dtype=np.int32) # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "imgs/bags/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (50, 50))
		x_bags.append(hog.compute(img, (64, 64)))
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = np.array(2 * np.ones(train_bags.shape[0], dtype=np.int32),dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))
	# labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
	# labels = np.array(labels)

	# labels转换成10行1列的矩阵
	# labels = labels.reshape(10, 1)
	# trainingData转换成10行2列的矩阵
	# trainingData = trainingData.reshape(10, 2)

	# 创建分类器
	ann =AnnClass(trainingData,labels)

	ret = ann.train()

	im = cv2.imread('imgs/test/bag1.bmp')
	# 预处理部分
	colorlow = (61, 83, 31)
	colorhigh = (81, 255, 250)
	colormin, colormax = np.array(colorlow), np.array(colorhigh)
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	# 去除颜色范围外的其余颜色
	mask = cv2.inRange(hsv, colormin, colormax)
	ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
	# 去噪
	binary = cv2.medianBlur(binary, 3)

	cv2.namedWindow("binary", 0)
	cv2.imshow("binary", binary)

	contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	moderatesize_countours = []
	moderatesize_rects = []
	boxindex = 0
	for countour in contours:
		countour_rect = cv2.boundingRect(countour)
		rect_x, rect_y, rect_w, rect_h = countour_rect
		if rect_w > 25 and rect_h > 25 and cv2.contourArea(countour) > 300:
			moderatesize_rects.append(countour_rect)
			moderatesize_countours.append(countour)

			testimg = im[rect_y + 1:rect_h + rect_y, rect_x + 1:rect_w + rect_x]
			testimg = cv2.resize(testimg, (50, 50))
			testfeature = hog.compute(testimg, (64, 64))
			testfeature = np.array([testfeature], dtype=np.float32)
			(ret, res) = ann.predictData(testfeature)
			print(ret, str(int(res[0][0])))
			cv2.putText(im, "res:{}".format(str(int(res[0][0]))),
			            (int(rect_x + 0.5 * rect_w + 50), int(rect_y + 0.5 * rect_h + 50)),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

	# for feature, rect in zip(testfeatures, rects):
	# 	pass

	cv2.drawContours(im, moderatesize_countours, -1, (0, 255, 0), 1)

	cv2.namedWindow("im", 0)
	cv2.imshow("im", im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# from objectdetect1 import hog
def hug_svm_test():
	# 效果较好
	win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
	block_size = (16, 16)  # （每个块最大为16 * 16）
	block_stride = (8, 8)  # 单元格尺寸
	cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
	num_bins = 10  # 对于每一个单元格，统计9个方向的梯度直方图。
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
	land_mark_dir = "C:/work/imgs/land_mark"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "{dir}/{i}".format(dir=land_mark_dir, i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		x_lands.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.ones(train_lands.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos

	othersdir = "C:/work/imgs/others/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_ohters.append(hog.compute(img, (64, 64)))
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = 3 * np.ones(train_ohers.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "C:/work/imgs/bags/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_bags.append(hog.compute(img, (64, 64)))
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = 2 * np.ones(train_bags.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))
	# labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
	# labels = np.array(labels)

	# labels转换成10行1列的矩阵
	# labels = labels.reshape(10, 1)
	# trainingData转换成10行2列的矩阵
	# trainingData = trainingData.reshape(10, 2)

	# 创建分类器
	svm = cv2.ml.SVM_create()
	# 设置svm类型
	svm.setType(cv2.ml.SVM_C_SVC)
	# 核函数
	svm.setKernel(cv2.ml.SVM_LINEAR)
	# 训练
	ret = svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

	# 测试数据
	# 取0-10之间的整数值
	# arrayTest = np.empty(shape=[0, 2], dtype='float32')
	# for i in range(10):
	# 	for j in range(10):
	# 		arrayTest = np.append(arrayTest, [[i, j]], axis=0)
	# pt = np.array(np.random.rand(50, 2) * 10, dtype='float32')  # np.random.rand(50,2) * 10可以替换成arrayTest

	filename = "C:/work/imgs/test/10.png"
	testimg = cv2.imread(filename)
	cv2.imshow("test", testimg)
	testimg = cv2.resize(testimg, (512, 512))
	testfeature = hog.compute(testimg, (64, 64))
	testfeature = np.array([testfeature], dtype=np.float32)
	# 预测
	(ret, res) = svm.predict(testfeature)
	print(ret, res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def org_svm_test():
	# ValueError: setting an array element with a sequence.
	orb = cv2.ORB_create()
	# 检测关键点和特征描述

	land_mark_dir = "land_imgs/"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "land_imgs/{i}".format(i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		keypoint1, desc1 = orb.detectAndCompute(img, None)
		print(desc1)
		x_lands.append(desc1)  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.ones(train_lands.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos

	othersdir = "ohers_img/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		keypoint1, desc1 = orb.detectAndCompute(img, None)
		x_ohters.append(desc1)  # 利用HOG进行计算
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = -np.ones(train_ohers.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "bags_img/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		keypoint1, desc1 = orb.detectAndCompute(img, None)
		x_bags.append(desc1)
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = 2 * np.ones(train_bags.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))
	# labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
	# labels = np.array(labels)

	# labels转换成10行1列的矩阵
	# labels = labels.reshape(10, 1)
	# trainingData转换成10行2列的矩阵
	# trainingData = trainingData.reshape(10, 2)

	# 创建分类器
	svm = cv2.ml.SVM_create()
	# 设置svm类型
	svm.setType(cv2.ml.SVM_C_SVC)
	# 核函数
	svm.setKernel(cv2.ml.SVM_LINEAR)
	# 训练
	ret = svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

	# 测试数据
	# 取0-10之间的整数值
	# arrayTest = np.empty(shape=[0, 2], dtype='float32')
	# for i in range(10):
	# 	for j in range(10):
	# 		arrayTest = np.append(arrayTest, [[i, j]], axis=0)
	# pt = np.array(np.random.rand(50, 2) * 10, dtype='float32')  # np.random.rand(50,2) * 10可以替换成arrayTest

	filename = "red_test.png"
	testimg = cv2.imread(filename)
	cv2.imshow("test", testimg)
	testimg = cv2.resize(testimg, (512, 512))
	keypoint1, desc1 = orb.detectAndCompute(testimg, None)
	# testfeature = hog.compute(testimg, (64, 64))
	testfeature = np.array([desc1], dtype=np.float32)
	# 预测
	(ret, res) = svm.predict(testfeature)
	print(ret, res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def hug_knn_test():
	# 效果很差
	win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
	block_size = (16, 16)  # （每个块最大为16 * 16）
	block_stride = (8, 8)  # 单元格尺寸
	cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
	num_bins = 10  # 对于每一个单元格，统计9个方向的梯度直方图。
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

	land_mark_dir = "land_imgs/"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "land_imgs/{i}".format(i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		x_lands.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.ones(train_lands.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos

	othersdir = "ohers_img/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_ohters.append(hog.compute(img, (64, 64)))
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = -np.ones(train_ohers.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "bags_img/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_bags.append(hog.compute(img, (64, 64)))
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = 2 * np.ones(train_bags.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))
	knn = cv2.ml.KNearest_create()
	knn.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

	filename = "bag_test1.png"
	testimg = cv2.imread(filename)
	cv2.imshow("test", testimg)
	testimg = cv2.resize(testimg, (512, 512))
	testfeature = hog.compute(testimg, (64, 64))
	testfeature = np.array([testfeature], dtype=np.float32)

	ret, result, neighbours, dist = knn.findNearest(testfeature, k=5)
	print(ret)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def hub_bys_test():
	# 结果莫名其妙
	win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
	block_size = (16, 16)  # （每个块最大为16 * 16）
	block_stride = (8, 8)  # 单元格尺寸
	cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
	num_bins = 10  # 对于每一个单元格，统计9个方向的梯度直方图。
	hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

	land_mark_dir = "land_imgs/"  # 设置正负数据集所在的位置
	x_lands = []
	for landfile in os.listdir(land_mark_dir):  # 在0到900张图片中随机挑选400张图片
		filename = "land_imgs/{i}".format(i=landfile)
		print(filename)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		if img is None:
			print('Could not find image %s' % filename)
			continue
		x_lands.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算
	train_lands = np.array(x_lands, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	label_lands = np.ones(train_lands.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos

	othersdir = "ohers_img/"
	x_ohters = []
	for otherfile in os.listdir(othersdir):
		filename = '%s/%s' % (othersdir, otherfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_ohters.append(hog.compute(img, (64, 64)))
	train_ohers = np.array(x_ohters, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_others = -np.ones(train_ohers.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	bagsdir = "bags_img/"
	x_bags = []
	for bagfile in os.listdir(bagsdir):
		filename = '%s/%s' % (bagsdir, bagfile)
		img = cv2.imread(filename)
		img = cv2.resize(img, (512, 512))
		x_bags.append(hog.compute(img, (64, 64)))
	train_bags = np.array(x_bags, dtype=np.float32)  # 数据类型转换，兼容OpenCv
	lable_bags = 2 * np.ones(train_bags.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1

	# 将X 和Y （正样本和负样本）合并
	# x = np.concatenate((x_pos, x_neg))
	# y = np.concatenate((y_pos, y_neg))
	trainingData = np.concatenate((train_lands, train_ohers, train_bags))  # 数据集一定要设置成浮点型

	# trainingData = np.array(
	# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
	# 	dtype='float32');  # 数据集一定要设置成浮点型
	labels = np.concatenate((label_lands, lable_others, lable_bags))

	filename = "bag_test1.png"
	testimg = cv2.imread(filename)
	testimg = cv2.resize(testimg, (512, 512))
	testfeature = hog.compute(testimg, (64, 64))
	testfeature = np.array(testfeature, dtype=np.float32)

	trainingData = trainingData.reshape((-1, 1))
	# testfeature=testfeature.reshape(-1)
	# labels=labels.reshape(-1)
	print(trainingData.shape, testfeature.shape, labels.shape)
	# testfeature=np.concatenate(testfeature).reshape((-1,1))
	from sklearn import datasets
	iris = datasets.load_iris()
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	length = min(len(trainingData), len(labels))
	clf = clf.fit(trainingData[0:length], labels[0:length])

	y_pred = clf.predict(testfeature)
	print(y_pred)


# print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))


def sift_test():
	imgname1 = 'without_person.bmp'
	imgname2 = 'have_person.bmp'

	sift = cv2.xfeatures2d.SIFT_create()

	# FLANN 参数设计
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	img1 = cv2.imread(imgname1)
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
	kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

	img2 = cv2.imread(imgname2)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# hmerge = np.hstack((gray1, gray2))  # 水平拼接
	# cv2.imshow("gray", hmerge)  # 拼接显示为gray
	# cv2.waitKey(0)

	# img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
	# img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))
	#
	# hmerge = np.hstack((img3, img4))  # 水平拼接
	# cv2.imshow("point", hmerge)  # 拼接显示为gray
	# cv2.waitKey(0)
	matches = flann.knnMatch(des1, des2, k=2)
	matchesMask = [[0, 0] for i in range(len(matches))]

	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append([m])

	img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
	cv2.namedWindow("FLANN", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("FLANN", img5)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def surf_test():
	imgname1 = 'img1.png'
	imgname2 = 'img.png'

	sift = cv2.xfeatures2d.SURF_create()

	# FLANN 参数设计
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	img1 = cv2.imread(imgname1)
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
	kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

	img2 = cv2.imread(imgname2)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# hmerge = np.hstack((gray1, gray2))  # 水平拼接
	# cv2.imshow("gray", hmerge)  # 拼接显示为gray
	# cv2.waitKey(0)

	# img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
	# img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))
	#
	# hmerge = np.hstack((img3, img4))  # 水平拼接
	# cv2.imshow("point", hmerge)  # 拼接显示为gray
	# cv2.waitKey(0)
	matches = flann.knnMatch(des1, des2, k=2)
	matchesMask = [[0, 0] for i in range(len(matches))]

	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append([m])

	img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
	cv2.namedWindow("FLANN", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("FLANN", img5)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def orb_test():
	# 按照灰度图像读入两张图片
	img1 = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)
	img2 = cv2.imread("img1.png", cv2.IMREAD_GRAYSCALE)

	# 获取特征提取器对象
	orb = cv2.ORB_create()
	# 检测关键点和特征描述
	keypoint1, desc1 = orb.detectAndCompute(img1, None)
	keypoint2, desc2 = orb.detectAndCompute(img2, None)
	"""
	keypoint 是关键点的列表
	desc 检测到的特征的局部图的列表
	"""
	# 获得knn检测器
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.knnMatch(desc1, desc2, k=1)
	"""
	knn 匹配可以返回k个最佳的匹配项
	bf返回所有的匹配项
	"""
	# 画出匹配结果
	img3 = cv2.drawMatchesKnn(img1, keypoint1, img2, keypoint2, matches, img2, flags=2)
	cv2.namedWindow("matches", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("matches", img3)
	cv2.waitKey()
	cv2.destroyAllWindows()


def roi_test():
	# im = cv2.imread('img.png')
	im = cv2.imread('have_person.bmp')
	x, y, w, h = 651, 540, 94, 88
	roi_img = im[y + 1:y + h, x + 1:x + w]
	cv2.imshow("1", roi_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def hog_svm_detect():
	'''
	HOG检测人
	'''

	def is_inside(o, i):
		'''
		判断矩形o是不是在i矩形中
		args:
			o：矩形o  (x,y,w,h)
			i：矩形i  (x,y,w,h)
		'''
		ox, oy, ow, oh = o
		ix, iy, iw, ih = i
		return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

	def draw_person(img, person):
		'''
		在img图像上绘制矩形框person
		args:
			img：图像img
			person：人所在的边框位置 (x,y,w,h)
		'''
		x, y, w, h = person
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

	def detect_test():
		'''
		检测人
		'''
		img = cv2.imread('imgs/dog_person.jpg')
		rows, cols = img.shape[:2]
		sacle = 0.5
		print('img', img.shape)
		img = cv2.resize(img, dsize=(int(cols * sacle), int(rows * sacle)))
		# print('img',img.shape)
		# img = cv2.resize(img, (128,64))

		# 创建HOG描述符对象
		# 计算一个检测窗口特征向量维度：(64/8 - 1)*(128/8 - 1)*4*9 = 3780
		'''
		winSize = (64,128)
		blockSize = (16,16)    
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9    
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)  
		'''
		hog = cv2.HOGDescriptor()
		# hist = hog.compute(img[0:128,0:64])   计算一个检测窗口的维度
		# print(hist.shape)
		detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
		print('detector', type(detector), detector.shape)
		hog.setSVMDetector(detector)

		# 多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
		found, w = hog.detectMultiScale(img)
		# print('found', type(found), found.shape)

		# 过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
		found_filtered = []
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				# r在q内？
				if ri != qi and is_inside(r, q):
					break
			else:
				found_filtered.append(r)

		for person in found_filtered:
			draw_person(img, person)

		cv2.imshow('img', img)
		cv2.waitKey()
		cv2.destroyAllWindows()

	detect_test()


def fast_detect():
	img = cv2.imread('without_person.bmp')

	# Initiate FAST object with default values
	fast = cv2.FastFeatureDetector()

	# find and draw the keypoints
	kp = fast.detect(img, None)
	img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

	# Print all default params
	print("Threshold: ", fast.getInt('threshold'))
	print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
	print("neighborhood: ", fast.getInt('type'))
	print("Total Keypoints with nonmaxSuppression: ", len(kp))

	cv2.imshow('fast_true.png', img2)

	# Disable nonmaxSuppression
	fast.setBool('nonmaxSuppression', 0)
	kp = fast.detect(img, None)

	print("Total Keypoints without nonmaxSuppression: ", len(kp))

	img3 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

	cv2.imshow('fast_false.png', img3)
	cv2.waitKey()
	cv2.destroyAllWindows()


def orb_detect():
	import cv2
	# from matplotlib import pyplot as plt

	img = cv2.imread('without_person.bmp')
	cv2.imshow("img", img)

	# Initiate STAR detector
	orb = cv2.ORB()

	# find the keypoints with ORB
	kp = orb.detect(img, None)

	# compute the descriptors with ORB
	kp, des = orb.compute(img, kp)

	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)
	# plt.imshow(img2), plt.show()
	cv2.imshow("img2", img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def orb_match():
	img1 = cv2.imread("img1.png")  # 导入灰度图像
	img2 = cv2.imread("img.png")

	detector = cv2.ORB_create()

	kp1 = detector.detect(img1, None)
	kp2 = detector.detect(img2, None)
	kp1, des1 = detector.compute(img1, kp1)
	kp2, des2 = detector.compute(img2, kp2)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	# img3 = drawMatches(img1, kp1, img2, kp2, matches[:50])
	# # img3 = cv2.drawKeypoints(img1,kp,None,color = (0,255,0),flags = 0)
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=0)
	# cv2.imwrite("orbTest.jpg", img3)
	cv2.imshow('orbTest', img3)
	cv2.waitKey(0)


def car_detect():
	import cv2
	import numpy as np
	# 此数据集为UIUC Car Detection 可网上下载
	datapath = "E:/迅雷下载/CarData/CarData/TrainImages"

	def path(cls, i):
		return "%s/%s%d.pgm" % (datapath, cls, i + 1)

	pos, neg = "pos-", "neg-"  # 数据集中图片命名方式

	detect = cv2.xfeatures2d.SIFT_create()  # 提取关键点
	# detect=cv2.ORB_create()
	extract = cv2.xfeatures2d.SIFT_create()  # 提取特征
	# extract=cv2.ORB_create()
	# FLANN匹配器有两个参数：indexParams和searchParams,以字典的形式进行参数传参
	flann_params = dict(algorithm=1, trees=5)  # 1为FLANN_INDEX_KDTREE
	matcher = cv2.FlannBasedMatcher(flann_params, {})  # 匹配特征
	# 创建bow训练器，簇数为40
	bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
	# 初始化bow提取器
	extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

	def extract_sift(fn):  # 参数为路径
		im = cv2.imread(fn, 0)
		return extract.compute(im, detect.detect(im))[1]  # 返回描述符

	# 读取8个正样本和8个负样本
	for i in range(8):
		bow_kmeans_trainer.add(extract_sift(path(pos, i)))
		bow_kmeans_trainer.add(extract_sift(path(neg, i)))
	# 利用训练器的cluster（）函数，执行k-means分类并返回词汇
	# k-means：属于聚类算法，所谓的聚类算法属于无监督学习，将样本x潜在所属类别Y找出来，具体稍后写一篇补上
	voc = bow_kmeans_trainer.cluster()
	extract_bow.setVocabulary(voc)

	def bow_features(fn):
		im = cv2.imread(fn, 0)
		return extract_bow.compute(im, detect.detect(im))

	# 两个数组，分别为训练数据和标签，并用bow提取器产生的描述符填充
	traindata, trainlabels = [], []
	for i in range(20):
		traindata.extend(bow_features(path(pos, i)));
		trainlabels.append(1)  # 1为正匹配
		traindata.extend(bow_features(path(neg, i)));
		trainlabels.append(-1)  # -1为负匹配
	# 创建SVM实例，将训练数据和标签放到numpy数组中进行训练，有关SVM知识稍后写一篇补上
	svm = cv2.ml.SVM_create()
	svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

	def predict(fn):
		f = bow_features(fn);
		p = svm.predict(f)
		print
		fn, "\t", p[1][0][0]
		return p

	# 预测结果
	car, notcar = "car.jpg", "imgs/dog_person.jpg"
	car_img = cv2.imread(car)
	notcar_img = cv2.imread(notcar)
	car_predict = predict(car)
	not_car_predict = predict(notcar)
	# 添加文字说明
	font = cv2.FONT_HERSHEY_SIMPLEX

	if (car_predict[1][0][0] == 1.0):  # predict结果为1.0表示能检测到汽车
		cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

	if (not_car_predict[1][0][0] == -1.0):  # predict结果为-1.0表示不能检测到汽车
		cv2.putText(notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('BOW + SVM Success', car_img)
	cv2.imshow('BOW + SVM Failure', notcar_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	im = cv2.imread('C:/work/imgs/test/bag5.bmp')
	# rectangle_detect()
	# # im = cv2.imread('img.png')
	a = PointLocationService(img=im)
	a.location_objects(flag=BAG_AND_LANDMARK)
	# hug_svm_detect_contours()
	# hug_ann_detect_contours()
	# hug_svm_test()
# orb_match()
# fast_detect()
# orb_test()
# car_detect()
# hug_svm_test()
# hug_knn_test()
# hub_bys_test()
# org_svm_test()

import cv2
import numpy as np
import os


class SvmUtil:

	def __init__(self, IMG_WIDTH=512, IMG_HEIGHT=512):
		# 创建分类器
		self.svm = cv2.ml.SVM_create()
		# 设置svm类型
		self.svm.setType(cv2.ml.SVM_C_SVC)
		# 核函数
		self.svm.setKernel(cv2.ml.SVM_LINEAR)
		self.IMG_WIDTH, self.IMG_HEIGHT = IMG_WIDTH, IMG_HEIGHT

	# ------------------------------------------------
	# 名称：init_hog
	# 功能：加载hog运算符,hog用来检测边缘特征
	# 参数： []
	# 返回： hog --- hog模型x
	# 作者：王杰     编写 ： 2020-3-xx    修改 ： 2020-8-12
	# ------------------------------------------------
	def init_hog(self):
		win_size = (48, 96)  # 检测窗口大小（检测对象的最小尺寸48 * 96）
		block_size = (16, 16)  # （每个块最大为16 * 16）
		block_stride = (8, 8)  # 单元格尺寸
		cell_size = (8, 8)  # （从一个单元格移动8*8像素到另外一个单元格）
		num_bins = 10  # 对于每一个单元格，统计9个方向的梯度直方图。
		hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
		return hog

	# ------------------------------------------------
	# 名称：load_ng_data
	# 功能：加载消极数据，目前使用的是预测图片数据
	# 参数： [hog]     ---hog算子
	#        [ng_dir]  ---消极图片路径
	# 返回： [None]
	# 作者：王杰     编写 ： 2020-8-12
	# ------------------------------------------------
	def load_ng_imgs(self, hog, ng_dir):
		ng_samples = []
		for ngfilename in os.listdir(ng_dir):
			filename = '%s/%s' % (ng_dir, ngfilename)
			if '.txt' in filename:
				continue
			img = cv2.imread(filename)
			img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
			ng_samples.append(hog.compute(img, (64, 64)))
		ng_traindatas = np.array(ng_samples, dtype=np.float32)  # 数据类型转换，兼容OpenCv
		lable_others = -1 * np.ones(ng_traindatas.shape[0], dtype=np.int32)  # 将训练样本赋值给y_neg   np.ones 填充1
		return lable_others, ng_traindatas

	# ------------------------------------------------
	# 名称：load_pos_data
	# 功能：加载正面数据，目前使用的是预测图片数据
	# 参数： [hog]     ---hog算子
	#        [ng_dir]  ---消极图片路径
	# 返回： [None]
	# 作者：王杰     编写 ： 2020-8-12
	# ------------------------------------------------
	def load_pos_imgs(self, hog, pos_dir):
		pos_hogs = []
		for pos_filename in os.listdir(pos_dir):  # 在0到900张图片中随机挑选400张图片

			filename = "{dir}/{filename}".format(dir=pos_dir, filename=pos_filename)
			if '.txt' in filename:
				continue
			print(filename)
			img = cv2.imread(filename)
			img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
			if img is None:
				print('Could not find image %s' % filename)
				continue
			pos_hogs.append(hog.compute(img, (64, 64)))  # 利用HOG进行计算,梯度特征
		train_datas = np.array(pos_hogs, dtype=np.float32)  # 数据类型转换，兼容OpenCv
		labels_pos = np.ones(train_datas.shape[0], dtype=np.int32)  # 将训练样本赋值为1给y_pos
		return labels_pos, train_datas

	# ------------------------------------------------
	# 名称：detect_ifhas_label
	# 功能：判断有无标签
	# 参数： [None]
	# 返回： int --- 1:正 -1 负
	# 作者：王杰     编写 ： 2020-8-12   修改 ：-
	# ------------------------------------------------
	def predict_img(self, ng_dir="D:/wj/px_train/neg", pos_dir="D:/wj/px_train/pos",
	                testimg_path="D:/wj/px_test/123/Image_20200622095609159.bmp", stand='upright'):

		hog = self.init_hog()

		# 加载正训练数据
		labels_pos, pos_traindatas = self.load_pos_imgs(hog, pos_dir)

		# 加载负训练数据
		lable_others, ng_traindatas = self.load_ng_imgs(hog, ng_dir)

		# 正样本和负样本合并

		trainingData = np.concatenate((pos_traindatas, ng_traindatas))  # 数据集一定要设置成浮点型

		# trainingData = np.array(
		# 	[[10, 3], [5, 0.5], [10, 5], [0.5, 10], [0.5, 1.6], [3, 6], [1.2, 4], [6, 6], [0.9, 5], [4, 4]],
		# 	dtype='float32');  # 数据集一定要设置成浮点型
		labels = np.concatenate((labels_pos, lable_others))
		# labels = (1, -1, 1, 1, -1, -1, -1, 1, -1, -1)
		# labels = np.array(labels)

		# labels转换成10行1列的矩阵
		# labels = labels.reshape(10, 1)
		# trainingData转换成10行2列的矩阵
		# trainingData = trainingData.reshape(10, 2)

		# # 创建分类器
		# svm = cv2.ml.SVM_create()
		# # 设置svm类型
		# svm.setType(cv2.ml.SVM_C_SVC)
		# # 核函数
		# svm.setKernel(cv2.ml.SVM_LINEAR)
		# 训练
		ret = self.svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)

		# 测试数据
		# 取0-10之间的整数值
		# arrayTest = np.empty(shape=[0, 2], dtype='float32')
		# for i in range(10):
		# 	for j in range(10):
		# 		arrayTest = np.append(arrayTest, [[i, j]], axis=0)
		# pt = np.array(np.random.rand(50, 2) * 10, dtype='float32')  # np.random.rand(50,2) * 10可以替换成arrayTest

		# filename = "D:/wj/px_test/123/Image_20200622095349271.bmp"
		# filename = "D:/wj/px_test/123/Image_20200622095609159.bmp"
		# filename = "D:/wj/px_test/Image_20200621160435205.bmp"
		testimg = cv2.imread(testimg_path)

		testimg = cv2.resize(testimg, (self.IMG_WIDTH, self.IMG_HEIGHT))
		testfeature = hog.compute(testimg, (64, 64))
		testfeature = np.array([testfeature], dtype=np.float32)
		# 预测
		(ret, res) = self.svm.predict(testfeature)
		print(ret, res)
		if stand == 'upright':
			cv2.putText(testimg, "{}".format("upright" if res == 1 else "handstand"),
			            (60, 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		else:
			cv2.putText(testimg, "{}".format("ok" if res == 1 else "ng"),
			            (60, 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		cv2.imshow("test", testimg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		return res


# 字体有可能上下颠倒，要做透视变换      通过训练判断是否倒立
# 检测是否“合格”两个字
# 如果包含，那就显示成功
# 如果失败，显示失败
# 不管有无标签，都要将显示的字符显示到 文本框种，并标注是否“合格”
def test_ok():
	# filename = "D:/wj/px_test/123/Image_20200622095609159.bmp"
	test_filename = "D:/wj/px_test/20200622_121041_2_0.bmp"
	testimg = cv2.imread(test_filename)
	testimg = cv2.resize(testimg, (512, 512))

	upright_flag = SvmUtil().predict_img(ng_dir="D:/wj/px_train/handstand", pos_dir="D:/wj/px_train/upright",
	                                     testimg_path=test_filename, stand="upright")

	if upright_flag == -1:
		# img = cv2.imread('messi5.jpg', 0)
		rows, cols, chanels = testimg.shape
		M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
		after_affinewarp = cv2.warpAffine(testimg, M, (cols, rows))
		# blurred = cv2.GaussianBlur(after_affinewarp, (3, 3), 0)
		gray = cv2.cvtColor(after_affinewarp, cv2.COLOR_RGB2GRAY)
		canny_img=cv2.Canny(gray,10,250)
		cv2.imshow("canny_img", canny_img)

	# cv2.putText(testimg, "{}".format("test"),
	#             (60, 60),
	#             cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
	# cv2.imshow("test", testimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	test_ok()
# SvmUtil().predict_img( ng_dir="D:/wj/px_train/neg", pos_dir="D:/wj/px_train/pos",
#                 testimg_path="D:/wj/px_test/123/Image_20200622095609159.bmp",stand="qualified")

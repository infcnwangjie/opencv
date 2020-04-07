# -*- coding: utf-8 -*-
import cv2
# from sklearn import svm
import numpy as np


class AnnClass(object):
	def __init__(self, traindatas, labels):
		animals_net = cv2.ml.ANN_MLP_create()

		# ANN_MLP_RPROP和ANN_MLP_BACKPROP都是反向传播算法，此处设置相应的拓扑结构
		animals_net.setLayerSizes(np.array([3, 6, 4]))
		animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
		animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)

		# 指定ANN的终止条件
		animals_net.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
		self.animals_net = animals_net
		self.traindatas = traindatas
		self.labels = labels

	def train(self):
		for train, label in zip(self.traindatas, self.labels):
			print(train)
			self.animals_net.train(train, cv2.ml.ROW_SAMPLE, label)

	def predict(self, testdata):
		return self.animals_net.predict(testdata, dtype=np.float32)


class SvmClass(object):
	def __init__(self, traindata, labels):
		# self.trainingData = traindata
		# self.svmObject = svm.SVC(kernel='rbf', gamma=50, C=1.0, probability=True)
		# 创建分类器
		self.svm = cv2.ml.SVM_create()
		# 设置svm类型
		self.svm.setType(cv2.ml.SVM_C_SVC)
		# 核函数
		self.svm.setKernel(cv2.ml.SVM_LINEAR)
		self.traindata, self.labels = traindata, labels

	def trainData(self):
		# self.svmObject.fit(self.trainingData[0], self.trainingData[1])
		# 训练
		ret = self.svm.train(self.traindata, cv2.ml.ROW_SAMPLE, self.labels)

	def predictData(self, testdata):
		# predictedValue = self.svmObject.predict_proba(testdata)
		(ret, res) = self.svm.predict()
		return ret, res

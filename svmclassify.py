import cv2
# from sklearn import svm
import numpy as np


class svmClass(object):
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
		(ret, res) = self.svm.predict(testdata)
		return res



import cv2
import numpy
from PIL import Image
import pytesseract

from PIL import Image
import numpy as np






'''水平投影'''
def getHProjection(image):
	hProjection = np.zeros(image.shape, np.uint8)
	# 图像高与宽
	(h, w) = image.shape
	# 长度与图像高度一致的数组
	h_ = [0] * h
	# 循环统计每一行白色像素的个数
	for y in range(h):
		for x in range(w):
			if image[y, x] == 255:
				h_[y] += 1
	# 绘制水平投影图像
	for y in range(h):
		for x in range(h_[y]):
			hProjection[y, x] = 255
	cv2.imshow('hProjection2', hProjection)

	return h_

def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    cv2.imshow('vProjection',vProjection)
    return w_


def test_word_split():
	# 读入原始图像
	origineImage = cv2.imread('D:/wj/words.png')
	# 图像灰度化
	# image = cv2.imread('test.jpg',0)
	image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray', image)
	# 将图片二值化
	retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow('binary', img)
	# 图像高与宽
	(h, w) = img.shape
	Position = []
	# 水平投影
	H = getHProjection(img)

	start = 0
	H_Start = []
	H_End = []
	# 根据水平投影获取垂直分割位置
	for i in range(len(H)):
		if H[i] > 0 and start == 0:
			H_Start.append(i)
			start = 1
		if H[i] <= 0 and start == 1:
			H_End.append(i)
			start = 0
	# 分割行，分割之后再进行列分割并保存分割位置
	for i in range(len(H_Start)):
		# 获取行图像
		cropImg = img[H_Start[i]:H_End[i], 0:w]
		# cv2.imshow('cropImg',cropImg)
		# 对行图像进行垂直投影
		W = getVProjection(cropImg)
		Wstart = 0
		Wend = 0
		W_Start = 0
		W_End = 0
		for j in range(len(W)):
			if W[j] > 0 and Wstart == 0:
				W_Start = j
				Wstart = 1
				Wend = 0
			if W[j] <= 0 and Wstart == 1:
				W_End = j
				Wstart = 0
				Wend = 1
			if Wend == 1:
				Position.append([W_Start, H_Start[i], W_End, H_End[i]])
				Wend = 0
	# 根据确定的位置分割字符
	for m in range(len(Position)):
		cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238),
		              1)
	cv2.imshow('image', origineImage)
	cv2.waitKey(0)

def test_word_ocr():
	# text = pytesseract.image_to_string(Image.open(r'D:\wj\google.PNG'))
	img=cv2.imread("D:/wj/px_test/Image_20200621160435205.bmp")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	retval, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	(h, w) = img.shape
	M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -5, 1)
	after_affinewarp = cv2.warpAffine(img, M, (w, h))
	# cv2.imwrite("")
	cv2.imshow("dest",after_affinewarp[100:600,300:750])
	cv2.waitKey(0)

	text1 = pytesseract.image_to_string(after_affinewarp[100:600,300:750], lang='chi_sim')
	print(text1)



test_word_ocr()
# test_word_split()
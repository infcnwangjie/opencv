import cv2
import numpy as np


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [128], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [128], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def img_hist( color=[255, 0, 0]):
	image1=cv2.imread("C:/work/imgs/test/7.jpg")
	img1=cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
	# image2=cv2.imread("C:/work/imgs/test/test_landmark.png")
	image2=cv2.imread("C:/work/imgs/test/2.jpg")
	img2=cv2.cvtColor(image2,cv2.COLOR_BGR2HSV)
	# hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
	# minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
	# histImg = np.zeros([256, 256, 3], np.uint8)
	# hpt = int(0.9 * 256)
	i=calculate(img1,img2)
	print("相似度为:{}".format(i))

	cv2.namedWindow("image1")
	cv2.imshow("image1", image1)
	cv2.namedWindow("image2")
	cv2.imshow("image2", image2)


	cv2.waitKey(0)
	cv2.destroyAllWindows()

def sift_match():
	img1_gray = cv2.imread("../imgs/test/19.png")
	img2_gray = cv2.imread("../imgs/test/bag6.bmp")

	sift = cv2.xfeatures2d.SIFT_create()
	# sift = cv2.SURF()

	kp1, des1 = sift.detectAndCompute(img1_gray, None)
	kp2, des2 = sift.detectAndCompute(img2_gray, None)

	bf = cv2.BFMatcher(cv2.NORM_L2)
	matches = bf.knnMatch(des1, des2, k=2)

	goodMatch = []
	for m, n in matches:
		if m.distance < 0.50 * n.distance:
			goodMatch.append(m)
	res = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, goodMatch[:], None, flags=2)
	cv2.imshow('res', res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def orb_match():
	img1 = cv2.imread("../imgs/test/19.png")  # 导入灰度图像
	img2 = cv2.imread("../imgs/test/bag6.bmp")

	detector = cv2.ORB_create()

	kp1 = detector.detect(img1, None)
	kp2 = detector.detect(img2, None)
	kp1, des1 = detector.compute(img1, kp1)
	kp2, des2 = detector.compute(img2, kp2)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	for matchitem in matches:
		print(matchitem)
	# cv2.imshow("t",des2)
	# matches = sorted(matches, key=lambda x: x.distance)  # 据距离来排序
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=0)
	cv2.namedWindow("orbTest",0)
	cv2.imshow('orbTest', img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	# orb_match()
	# sift_match()
	img_hist()
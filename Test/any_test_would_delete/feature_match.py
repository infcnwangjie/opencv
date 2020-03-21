import cv2
import numpy as np


def img_hist( color=[255, 0, 0]):
	image=cv2.imread("../imgs/test/6.png")
	img=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
	histImg = np.zeros([256, 256, 3], np.uint8)
	hpt = int(0.9 * 256)

	cv2.namedWindow("image")
	cv2.imshow("image", image)

	for h in range(256):
		intensity = int(hist[h] * hpt / maxVal)
		cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
	cv2.namedWindow("histImg")
	cv2.imshow("histImg",histImg)

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
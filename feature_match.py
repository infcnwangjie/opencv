import cv2


def orb_match():
	img1 = cv2.imread("imgs/test/bag1.bmp")  # 导入灰度图像
	img2 = cv2.imread("imgs/test/bag2.bmp")

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

orb_match()
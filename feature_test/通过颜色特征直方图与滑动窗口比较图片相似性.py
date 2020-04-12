# -*- coding: utf-8 -*-
import cv2
#https://baijiahao.baidu.com/s?id=1615404760897105428&wfr=spider&for=pc

def color_similar_ratio(image1, image2):
	if image1 is None or image2 is None:
		return 0
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	# cv2.imshow("hist1",hist1)
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
	print(degree)
	# if degree > 0.56:
	# 	backproject = cv2.calcBackProject([img2], [0, 1], hist1, [0, 180, 0, 255.0], 1)
	# 	cv2.imshow("backproject", backproject)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()
	return degree


def slide():
	img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rows, cols = gray.shape
	for row in range(0,rows):
		for col in range(502, 612):
			# print("-" * 1000)
			yield (col, row, img[row:row + 80, col:col + 80])
		# for col in range(2619, 2743):
		# 	print("-" * 1000)
		# 	yield (col, row, img[row:row + 80, col:col + 80])




def my_testslide():
	roi_red_img=cv2.imread("D:/roi_red.png")
	for col,row,img in slide():
		# print("+"*100)
		# print("rows:{},cols:{}".format(row,col))
		roi_red_img=cv2.resize(roi_red_img,(80,80))
		similar=color_similar_ratio(roi_red_img,img)
		# print("similar:{}".format(similar))
		if similar>0.85:
			print("find red landmark")
			cv2.namedWindow("roi", 0)
			cv2.imshow("roi", roi_red_img)
			cv2.namedWindow("target")
			cv2.imshow("target",img)

		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == '__main__':
	# image1 = cv2.imread("D:/roi1.png")
	# image2 = cv2.imread("D:/target_gy.png")
	# i = color_similar_ratio(image1, image2)
	# print("color,相似度为:{}".format(i))
	my_testslide()

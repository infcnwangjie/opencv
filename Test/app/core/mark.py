import cv2

from app.config import RECT_TEMPLATE_PATH


def isRectange(contour):
	img = cv2.imread(RECT_TEMPLATE_PATH, 0)
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh, 3, 2)
	print(len(contours))
	result = cv2.drawContours(img, contours, -1,
	                          (0, 255, 0), 2)
	cv2.imshow("testrectang",result)
	rectange_contor = contours[2]
	rectange_diff = abs(cv2.matchShapes(contour, rectange_contor, 1, 0.0))
	return rectange_diff<0.4


def calculate(image1, image2):
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img1], [0], None, [128], [0.0, 255.0])
	hist2 = cv2.calcHist([img2], [0], None, [128], [0.0, 255.0])
	degree = 0
	for i in range(len(hist1)):
		if hist1[i] != hist2[i]:
			degree = degree + \
			         (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
		else:
			degree = degree + 1
	degree = degree / len(hist1)
	return degree


if __name__ == '__main__':
	image1 = cv2.imread("C:/work/imgs/test/template1.png")
	image2 = cv2.imread("C:/work/imgs/test/8.png")
	i = calculate(image1, image2)
	print("相似度为:{}".format(i))

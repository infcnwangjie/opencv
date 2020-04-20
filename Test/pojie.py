import cv2


def break_capcha():
	imgpath="D:/test.png"
	img=cv2.imread(imgpath)
	img=cv2.resize(img,(325,201))
	cv2.namedWindow("img")
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


break_capcha()
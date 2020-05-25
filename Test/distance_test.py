import math

import cv2

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.processers.bag_detector import BagDetector
from app.core.processers.landmark_detector import LandMarkDetecotr, WITH_TRANSPORT


# TODO 计算物体距离摄像头的距离
def compute_height(W=110, F=6, P=10, H=8000):
	'''
	用于求解袋子高度
	@:param W: 实际袋子条宽度
	@:param F:焦距                   已知项，摄像头焦距 F  6mm
	@:param P:图像像素宽度
	@:param H:摄像头距离地面的高度   已知项，摄像头高度 H　８米
	:return: D=H-W*F/P
	'''
	if P < 0:
		raise Exception("图像像素宽度计算有误")
	try:
		d= W * F / P
	except:
		raise Exception("计算有误")
	return d


def compute_x(L=10, H=8000, F=6, W=100, P=10, D=0):
	'''
	# TODO 计算物体X轴坐标值
	# X*X=L*L -(H-FW/P)*(H-FW/P)
	:param image:
	:return:
	'''
	try:
		a = math.pow(L, 2)
		b = D if D!=0 or D is not None else math.pow(H - F * W / P, 2)
		x = math.sqrt(abs(a - b))
	except Exception as e:
		raise e
	return x


def test_one_image():
	a = LandMarkDetecotr()
	# cap = cv2.VideoCapture("C:/NTY_IMG_PROCESS/VIDEO/Video_20200519095438779.avi")
	# cap.set(cv2.CAP_PROP_POS_FRAMES, 243)
	# ret, frame = cap.read()
	# frame=cv2.resize(frame,(IMG_HEIGHT,IMG_WIDTH))
	# cv2.namedWindow("src")
	# cv2.imshow("src",frame)
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152805570.bmp")
	image = cv2.imread("c:/work/nty/hangche/Image_20200522152729727.bmp")
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152825743.bmp")
	# image=cv2.imread("c:/work/nty/hangche/Image_20200522154907736.bmp")
	image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
	cv2.namedWindow("src")
	cv2.imshow("src", image)
	dest, success = a.position_landmark(image)
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)
	# src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-14-12-50-58test.bmp')).position_landmark()
	b = BagDetector()
	if WITH_TRANSPORT and success:
		for bag in b.location_bags(dest, middle_start=100, middle_end=400):
			D=compute_height(W=110,F=6,P=bag.width,H=9577)
			print("-" * 100)
			print("高度为：{}".format(D))
			print(bag)
			print("计算坐标：{}".format(compute_x(bag.x * 10, H=9577, F=6, W=110, P=bag.width,D=0)))
		a.draw_grid_lines(dest)
	else:
		for bag in b.location_bags(dest, middle_start=400, middle_end=600):
			print(bag)
		# print("计算坐标：{}".format(compute_x(bag.x * 10, H=8000, F=6, W=110, P=bag.width)))
	# print(dest.shape)

	# cap.release()
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	test_one_image()

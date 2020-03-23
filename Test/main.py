# encoding:utf-8
import os



import cv2
import numpy as np


if __name__ == '__main__':
	im = cv2.imread('C:/work/imgs/test/bag6.bmp')
	# rectangle_detect()
	# # im = cv2.imread('img.png')
	a = PointLocationService(img=im)
	a.location_objects(flag=BAG_AND_LANDMARK)
	# hug_svm_detect_contours()
	# hug_ann_detect_contours()
	# hug_svm_test()
# orb_match()
# fast_detect()
# orb_test()
# car_detect()
# hug_svm_test()
# hug_knn_test()
# hub_bys_test()
# org_svm_test()

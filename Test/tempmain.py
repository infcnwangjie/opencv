# encoding:utf-8
import os

import cv2

from app.core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK

if __name__ == '__main__':
	im = cv2.imread('C:/work/imgs/test/bag4.bmp')
	with PointLocationService(img=im,print_or_no=True) as  a:
		a.location_objects(flag=BAG_AND_LANDMARK)
		a.next_move()

# encoding:utf-8
import cv2

from core.target_detect.pointlocation import PointLocationService, BAG_AND_LANDMARK

if __name__ == '__main__':
	im = cv2.imread('C:/work/imgs/test/bag6.bmp')
	a = PointLocationService(img=im)
	a.location_objects(flag=BAG_AND_LANDMARK)

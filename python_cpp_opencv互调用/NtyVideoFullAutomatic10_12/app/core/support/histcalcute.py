# -*- coding: utf-8 -*-
import cv2
import numpy as np

def color_similar_ratio(image1, image2):
	if image1 is None or image2 is None:
		return 0
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
	return degree

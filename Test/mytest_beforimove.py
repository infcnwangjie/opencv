# -*- coding: utf-8 -*-
from itertools import chain

import cv2
import time

# https://baijiahao.baidu.com/s?id=1615404760897105428&wfr=spider&for=pc
from app.core.processers.preprocess import Preprocess

img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
dest = cv2.resize(img, (900, 700))
gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape

SLIDE_WIDTH = 30
SLIDE_HEIGHT = 30

FOND_RECT_WIDTH = 60
FOND_RECT_HEIGHT = 60


class SimilarSlideWindow:
	__slots__ = ['col', 'row', 'slide_img', 'similarity', 'times', 'window_info']

	def __init__(self, col, row, slide_img, similarity=0):
		self.col = col
		self.row = row
		self.slide_img = slide_img
		self.similarity = similarity
		self.times = 0

	@property
	def data(self):
		return self.col, self.row, self.slide_img


class TargetRect:
	'''已经识别到的rect'''

	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	def slider_in_rect(self, slide_obj: SimilarSlideWindow):
		slide_col, slide_row, _img = slide_obj.data
		slide_point1 = (slide_col, slide_row)
		slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)

		if self.point1[0] < slide_point1[0] and self.point1[1] < slide_point1[1] and self.point2[0] > slide_point2[
			0] and self.point2[1] > slide_point2[1]:
			return True
		return False


good_rects = []


class LandMarkRoi:
	def __init__(self, img, label):
		self.roi = img
		self.label = label
		self._best_similar_window = None
		self._best_similar_score = 0
		self._times = 0

	def add_match_slide(self, slide_window: SimilarSlideWindow):
		if slide_window.similarity > self._best_similar_score:
			self._best_similar_score
			self._best_similar_window = slide_window
			self._times += 1

	@property
	def times(self):
		return self._times

	@property
	def current_similarity(self):
		return self._best_similar_score

	@property
	def bset_match_window(self):
		return self._best_similar_window if self._best_similar_window else None


def tjtime(fun):
	def inner(*args, **kwargs):
		start = time.clock()
		result = fun(*args, **kwargs)
		end = time.clock()
		print("{}cost {}秒".format(fun.__name__, end - start))
		return result

	return inner


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print("x:{},y:{}".format(x, y))


# @tjtime
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

	return degree


def slide():
	global img, dest, gray, rows, cols
	for row in range(0, rows):
		for col in chain(range(156, 213), range(850, 890)):
			yield SimilarSlideWindow(col, row, dest[row:row + SLIDE_HEIGHT, col:col + SLIDE_WIDTH])


@tjtime
def my_testslide():
	global img, dest, good_rects
	landmark_rois = []
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/red.png"), label='red'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/greenyellow.png"), label='greenyellow'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/yellow_red.png"), label='yellow_red'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/red_green.png"), label='red_green'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/dark_red_green.png"), label='dark_red_green'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/dark_red_yellow.png"), label='dark_red_yellow'))
	landmark_rois.append(LandMarkRoi(img=cv2.imread("D:/dark_green_yellow.png"), label='dark_yellow_green'))

	not_hit_times = 0
	for slide_window_obj in slide():
		col, row, slide_img = slide_window_obj.data
		show_img = dest.copy()
		found = False
		for rect in good_rects:
			if rect.slider_in_rect(slide_window_obj):
				found = True
				break
		if found:
			continue

		# 迭代结束条件
		need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times ==0]

		if not_hit_times > 30000 or len(need_find_roi) == 0:
			print("not hit_times is {},need find roi is {}".format(not_hit_times,len(need_find_roi)))
			break

		cv2.rectangle(show_img, (col, row), (col + 30, row + 30), color=(255, 255, 0), thickness=2)
		hit = 0
		for landmark_roi in landmark_rois:
			if landmark_roi.times > 4:
				continue
			roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
			similar = color_similar_ratio(roi, slide_img)
			if similar > 0.60:
				hit += 1
				not_hit_times = 0
				print("find {} roi similar is {}".format(landmark_roi.label, similar))
				slide_window_obj.similarity = similar
				landmark_roi.add_match_slide(slide_window_obj)
				if landmark_roi.times > 5 or landmark_roi.current_similarity > 0.62:
					target_rect = TargetRect((col - FOND_RECT_WIDTH, row - FOND_RECT_HEIGHT),
					                         (col + FOND_RECT_WIDTH, row + FOND_RECT_HEIGHT))
					good_rects.append(target_rect)
				cv2.rectangle(show_img, (col, row), (col + SLIDE_WIDTH, row + SLIDE_HEIGHT), color=(255, 255, 0),
				              thickness=2)
				cv2.putText(show_img, "find {} roi similar is {}".format(landmark_roi.label, similar),
				            (col, row + 30),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
				cv2.namedWindow("target")
				cv2.imshow("target", show_img)
				cv2.waitKey(2000)
		if hit == 0:
			not_hit_times += 1

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	my_testslide()

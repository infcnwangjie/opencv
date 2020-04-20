# -*- coding: utf-8 -*-
from itertools import chain
from queue import Queue, LifoQueue
import cv2
import time

import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Thread, Lock
from multiprocessing import Lock as PLock
import numpy as np

# https://baijiahao.baidu.com/s?id=1615404760897105428&wfr=spider&for=pc
cv2.useOptimized()
img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
dest = cv2.resize(img, (900, 700))
# dest = cv2.resize(img, (1000, 800))

gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape

SLIDE_WIDTH = 20
SLIDE_HEIGHT = 20

FOND_RECT_WIDTH = 60
FOND_RECT_HEIGHT = 60

# tasks = Queue()
good_rects = []
results = Queue()
step = 1
fail_time = 0


def tjtime(fun):
	def inner(*args, **kwargs):
		start = time.clock()
		result = fun(*args, **kwargs)
		end = time.clock()
		print("{}cost {}秒".format(fun.__name__, end - start))
		return result

	return inner


class NearLandMark:
	__slots__ = ['col', 'row', '_slide_img', '_similarity', '_roi', 'direct']

	def __init__(self, col, row, slide_img, similarity=0):
		self.col = col
		self.row = row
		self._slide_img = slide_img
		self._similarity = similarity
		self._roi = None
		self.direct = 'left' if self.col < 0.5 * cols else 'right'  # 0 :L 1:R

	@property
	def data(self):
		return self.col, self.row, self._slide_img

	@property
	def similarity(self):
		return self._similarity

	@similarity.setter
	def similarity(self, value):
		self._similarity = value

	@property
	def roi(self):
		return self._roi

	@roi.setter
	def roi(self, value):
		value.times += 1
		self._roi = value


class TargetRect:
	'''已经识别到的rect'''

	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	# @tjtime
	def slider_in_rect(self, slide_obj: NearLandMark = None, slide_col=None, slide_row=None):

		if slide_col:
			slide_col, slide_row, _img = slide_obj.data
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		else:
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		print(self.point1[0], self.point2[0], slide_point1[0], slide_point1[1])
		if self.point1[0] < slide_point1[0] and self.point1[1] < slide_point1[1] and self.point2[0] > slide_point2[
			0] and self.point2[1] > slide_point2[1]:
			return True

		return False


class LandMarkRoi:

	def __init__(self, img, label, id=None):
		self.roi = img
		self.id = id
		self.label = label
		self._times = 0
		self.lock = Lock()

	@property
	def times(self):
		return self._times

	@times.setter
	def times(self, value):
		self.lock.acquire()
		self._times = value
		self.lock.release()


landmark_rois = [LandMarkRoi(img=cv2.imread("D:/red.png"), label='red2', id=1),
                 LandMarkRoi(img=cv2.imread("D:/greenyellow.png"), label='greenyellow', id=2),
                 LandMarkRoi(img=cv2.imread("D:/yellow_red.png"), label='yellow_red', id=3),
                 LandMarkRoi(img=cv2.imread("D:/red_green.png"), label='red_green', id=4),
                 LandMarkRoi(img=cv2.imread("D:/dark_red_green.png"), label='dark_red_green', id=5),
                 LandMarkRoi(img=cv2.imread("D:/dark_red_yellow.png"), label='dark_red_yellow', id=6),
                 LandMarkRoi(img=cv2.imread("D:/dark_green_yellow.png"), label='dark_yellow_green', id=7)]


# @tjtime
def compare_similar(img1, img2):
	if img1 is None or img2 is None:
		return 0
	hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
	cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
	degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
	return degree


def generator_slidewindows():
	global dest, rows, cols, step
	row = 0
	while row < rows:
		# for col in chain(range(156, 200), range(850, 890)):
		for col in chain(range(150, 189),range(766, 800)):
			for rect in good_rects:
				if rect.slider_in_rect(slide_col=col, slide_row=row):
					step=10
					break
			else:
				yield NearLandMark(col, row, dest[row:row + SLIDE_HEIGHT, col:col + SLIDE_WIDTH])
		if fail_time > 400:
			step += 1
		else:
			step = 1
		row += step


@tjtime
def computer_task(landmark_roi: LandMarkRoi, slide_window_obj):
	col, row, slide_img = slide_window_obj.data
	roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
	similar = compare_similar(roi, slide_img)
	global step, fail_time
	if similar > 0.60:
		slide_window_obj.similarity = similar
		slide_window_obj.roi = landmark_roi
		fail_time = 0
		good_rects.append(TargetRect((col - FOND_RECT_WIDTH,
		                              row - FOND_RECT_HEIGHT),
		                             (col + FOND_RECT_WIDTH,
		                              row + FOND_RECT_HEIGHT)))
		return slide_window_obj
	else:
		del slide_window_obj
		fail_time += 1
		# step += 1


# @tjtime
def main():
	start = time.clock()
	global img, dest, good_rects, slide_window_queue

	pool = ThreadPoolExecutor(7)
	for slide_window_obj in generator_slidewindows():
		# 迭代结束条件
		need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times == 0]
		if len(need_find_roi) == 0:
			print("need find roi is {}".format(len(need_find_roi)))
			break

		for landmark_roi in landmark_rois:
			future = pool.submit(computer_task, landmark_roi, slide_window_obj)
			results.put(future)

	pool.shutdown()

	last_window_tuple = None
	last_direct = None
	while results.qsize() > 0:
		future_obj = results.get()
		if future_obj.result() is None:
			continue

		slide_window_obj = future_obj.result()
		col = slide_window_obj.col
		row = slide_window_obj.row
		# if last_window_tuple is not None:
		# 	nearest = math.sqrt(math.pow(col - last_window_tuple[0], 2) + math.pow(
		# 		row - last_window_tuple[
		# 			1], 2), 2)
		# 	print("nearest{}:{}".format(slide_window_obj.roi.label, nearest))
		# 	if nearest < 60 and last_direct == slide_window_obj.direct:
		# 		last_window_tuple = (col, row)
		# 		last_direct = slide_window_obj.direct
		# 		continue
		# 	else:
		# 		last_window_tuple = (col, row)
		# 		last_direct = slide_window_obj.direct

		cv2.rectangle(dest, (col, row), (col + SLIDE_WIDTH, row + SLIDE_HEIGHT), color=(255, 255, 0),
		              thickness=1)
		cv2.putText(dest,
		            "{}:{}:{}".format(slide_window_obj.direct, landmark_roi.label,
		                              round(slide_window_obj.similarity, 2)),
		            (col, row + 30),
		            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 1)

	end = time.clock()
	print("结束{}".format(end - start))
	cv2.namedWindow("target")
	cv2.imshow("target", dest)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

# -*- coding: utf-8 -*-
from gevent import monkey;monkey.patch_all()


from itertools import chain
from queue import Queue, LifoQueue
import cv2
import time
import gevent

import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Thread, Lock
from multiprocessing import Lock as PLock
import numpy as np
# import asyncio
import profile


# 把程序变成协程的方式运行②


# https://baijiahao.baidu.com/s?id=1615404760897105428&wfr=spider&for=pc
cv2.useOptimized()
img = cv2.imread("D:/2020-04-10-15-26-22test.bmp")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
dest = cv2.resize(img, (900, 700))

gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape

SLIDE_WIDTH = 35
SLIDE_HEIGHT = 35

FOND_RECT_WIDTH = 90
FOND_RECT_HEIGHT = 90

tasks = Queue()
good_rects = []
results = Queue()
step = 2
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
	__slots__ = ['point1', 'point2']

	def __init__(self, point1, point2):
		self.point1 = point1
		self.point2 = point2

	def slider_in_rect(self, slide_obj: NearLandMark = None, slide_col=None, slide_row=None):
		if slide_obj:
			slide_col, slide_row, _img = slide_obj.data
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
		else:
			slide_point1 = (slide_col, slide_row)
			slide_point2 = (slide_col + SLIDE_WIDTH, slide_row + SLIDE_HEIGHT)
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
		# self.lock = Lock()
		self.land_marks = []

	def add_slide_window(self, slide_window: NearLandMark):
		# with self.lock:
		if len(self.land_marks) == 0:
			self.land_marks.append(slide_window)
		for land_mark in self.land_marks:
			col, row, similar = land_mark.col, land_mark.row, land_mark.similarity
			col1, row1, similar1 = slide_window.col, slide_window.row, slide_window.similarity
			if math.sqrt(math.pow(col - col1, 2) + math.pow(row - row1, 2)) < 50:
				if similar < similar1:
					del land_mark
				break
		else:
			self.land_marks.append(slide_window)

	# self.lock.release()

	@property
	def times(self):
		return self._times

	@times.setter
	def times(self, value):
		# self.lock.acquire()
		# with self.lock:
		self._times = value


# self.lock.release()


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
	# cols=list(chain(range(150, 175), range(766, 800)))
	while row < rows:
		for col in chain(range(150, 175), range(766, 796)):
			for rect in good_rects:
				if rect.slider_in_rect(slide_col=col, slide_row=row):
					break
			else:
				yield NearLandMark(col, row, dest[row:row + SLIDE_HEIGHT, col:col + SLIDE_WIDTH])
		if fail_time > 200:
			step += 1
		elif fail_time>10000:
			step+=300
		else:
			step = 2
		row += step


# @tjtime
def computer_task(landmark_roi: LandMarkRoi, slide_window_obj):
	# while tasks.qsize()>0:
	# slide_window_obj = tasks.get()
	col, row, slide_img = slide_window_obj.data
	roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
	similar = compare_similar(roi, slide_img)
	global step, fail_time
	if similar > 0.56:
		# print("find")
		slide_window_obj.similarity = similar
		slide_window_obj.roi = landmark_roi
		landmark_roi.add_slide_window(slide_window_obj)
		fail_time = 0
		good_rects.append(TargetRect((col - FOND_RECT_WIDTH,
		                              row - FOND_RECT_HEIGHT),
		                             (col + FOND_RECT_WIDTH,
		                              row + FOND_RECT_HEIGHT)))
		return slide_window_obj
	else:
		del slide_window_obj
		fail_time += 1


# @tjtime
def main():
	start = time.clock()
	global img

	# task_list = []
	for slide_window_obj in generator_slidewindows():
		# 迭代结束条件
		need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times == 0]
		if len(need_find_roi) == 0:
			print("need find roi is {}".format(len(need_find_roi)))
			break

		for landmark_roi in landmark_rois:
			task = gevent.spawn(computer_task, landmark_roi, slide_window_obj)
			task.join()

	for landmark_roi in landmark_rois:
		for slide_window_obj in landmark_roi.land_marks:
			col = slide_window_obj.col
			row = slide_window_obj.row

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
	# profile.run('main()')

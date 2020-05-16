# -*- coding: utf-8 -*-
# from gevent import monkey;
import os
import pickle
import random
from collections import defaultdict
from functools import cmp_to_key
import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, ROIS_DIR, LEFT_MARK_FROM, LEFT_MARK_TO, RIGHT_MARK_FROM, RIGHT_MARK_TO, \
	PROGRAM_DATA_DIR
# monkey.patch_all()
from itertools import chain
from queue import Queue, LifoQueue
import cv2
import time
import gevent
import math
import profile
# 把程序变成协程的方式运行②
from app.core.beans.models import LandMarkRoi, NearLandMark, TargetRect
from app.core.processers.bag_detector import BagDetector
from app.log.logtool import mylog_error
import re

cv2.useOptimized()

rows, cols = IMG_HEIGHT, IMG_WIDTH

SLIDE_WIDTH, SLIDE_HEIGHT = 25, 25

FOND_RECT_WIDTH, FOND_RECT_HEIGHT = 70, 70

LEFT_START, LEFT_END = 150, 175

RIGHT_START, RIGHT_END = 766, 796
# RIGHT_START, RIGHT_END = 490, 540
# tasks = Queue()
good_rects = []
step = 2
fail_time = 0

ALL_LANDMARKS_DICT = {}


def tjtime(fun):
	def inner(*args, **kwargs):
		start = time.clock()
		result = fun(*args, **kwargs)
		end = time.clock()
		print("{}cost {}秒".format(fun.__name__, end - start))
		return result

	return inner


def landmarkname_cmp(a, b):
	result_a = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', a[0])

	result_b = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', b[0])

	if result_a is None and result_b is None:
		return 0
	elif result_a is not None:
		return -1
	elif result_b is not None:
		return 1
	else:
		a_no = int(result_a.group(1))
		b_no = int(result_b.group(1))
		if a_no > b_no:
			return 1
		elif a_no == b_no:
			return 0
		else:
			return -1


class LandMarkDetecotr:
	landmarks = []

	def __init__(self, img):
		self.img = img
		self.compute_result_info = {}
		self.landmark_dic = {}

	def corners_levelfour(self, left_top_landmark_name):
		'''级别4获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=1)]
		return labels

	def corners_levelsix(self, left_top_landmark_name):
		'''级别6获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=2),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=2)]
		return labels

	def corners_leveleight(self, left_top_landmark_name):
		'''级别8获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=3),
		          self.__fetch_neigbbour(left_top_landmark_name, west_step=3)]
		return labels

	def __fetch_neigbbour(self, landmark_name, sourth_step: int = 0, west_step: int = 0):
		result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', landmark_name)
		if result is None:
			return landmark_name
		current_no = int(result.group(1))
		# current_direct = result.group(2)
		# next_direct = "R" if current_direct == "L" else "L"
		direct = "R" if sourth_step == 1 else "L"
		no = current_no + west_step if west_step > 0 else current_no
		landmark_labelname = "NO{NO}_{D}".format(NO=no, D=direct)
		return landmark_labelname

	def get_next_no(self, landmark_name, forward=False):
		result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))

		next_no = current_no + 1 if not forward else current_no - 1
		next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
		return next_landmark

	def get_opposite_landmark(self, landmark_name):
		import re
		result = re.match(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''', landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))
		current_d = result.group(2)
		opposite_d = 'R' if current_d == 'L' else 'L'
		next_landmark = "NO{NO}_{D}".format(NO=current_no, D=opposite_d)
		return next_landmark

	def compute_miss_landmark_position(self, landmark_name):
		global ALL_LANDMARKS_DICT
		samey_landmark = self.get_opposite_landmark(landmark_name)
		# print(samey_landmark)
		assert samey_landmark in ALL_LANDMARKS_DICT, "opposite landmark is not exist"

		samex_landmark = self.get_next_no(landmark_name)
		if samex_landmark not in ALL_LANDMARKS_DICT:
			samex_landmark = self.get_next_no(landmark_name, forward=True)
		try:
			x = ALL_LANDMARKS_DICT[samex_landmark].col
			y = ALL_LANDMARKS_DICT[samey_landmark].row
		except:
			raise Exception("landmark:{},same x is None".format(samex_landmark))
		return x, y

	def choose_best_cornors(self):
		'''本程序目前要求至少得获取三个角点，小于三个角点不支持。小于三个角点是没有任何意义的'''
		levels = ['4', '6']
		find = defaultdict(int)  # label1:0  label2:1
		loss = {}  # 闭合层级：丢失角点个数
		global ALL_LANDMARKS_DICT

		# print(ALL_LANDMARKS_DICT)
		for roi_item in self.__get_landmark_rois():
			label = roi_item.label
			if label in ALL_LANDMARKS_DICT:
				find[label] = 1
			else:
				find[label] = 0

		level_four_min_loss = 4
		level_six_min_loss = 6
		best_four_label_choose = None
		best_six_label_choose = None

		for roi_item in self.__get_landmark_rois():
			loss_info = {}
			for level in levels:
				label = roi_item.label
				if "_R" in label: continue
				if level == '4':
					point1, point2, point3, point4 = self.corners_levelfour(label)
					loss_info['4'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
					if loss_info['4'] < level_four_min_loss:
						best_four_label_choose = roi_item.label
						level_four_min_loss = loss_info['4']
				elif level == '6':
					point1, point2, point3, point4 = self.corners_levelsix(label)
					loss_info['6'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
					if loss_info['6'] < level_six_min_loss:
						best_six_label_choose = roi_item.label
						level_six_min_loss = loss_info['6']
				loss[label] = loss_info

		# print(ALL_LANDMARKS_DICT)
		positiondict = {}
		if loss[best_four_label_choose]['4'] < loss[best_six_label_choose]['6']:
			labels = self.corners_levelfour(best_four_label_choose)
			find_nums = sum([find[label] for label in labels])
			# print("best landmark is {},level four loss is {},level six loss is {}".format(best_four_label_choose,
			#                                                                               loss[best_four_label_choose][
			# 	                                                                              '4'],
			#                                                                               loss[best_four_label_choose][
			# 	                                                                              '6']))

		else:
			labels = self.corners_levelsix(best_six_label_choose)
			find_nums = sum([find[label] for label in labels])
			# assert find_nums >= 3, "landmark cornors must bigger than 3,only has {}".format(find_nums)
			# print("best landmark is {},level four loss is {},level six loss is {},and choose level 6".format(
			# 	best_six_label_choose,
			# 	loss[best_six_label_choose][
			# 		'4'],
			# 	loss[best_six_label_choose][
			# 		'6']))
		compensate_label = ""
		for label in labels:
			if label not in ALL_LANDMARKS_DICT:
				print("label {} need compute".format(label))
				compensate_label = label
				continue
			positiondict[label] = [ALL_LANDMARKS_DICT[label].col, ALL_LANDMARKS_DICT[label].row]
		if compensate_label != "":
			miss_x, miss_y = self.compute_miss_landmark_position(compensate_label)
			positiondict[compensate_label] = [miss_x, miss_y]
		print(positiondict)
		return positiondict

	def position_landmark(self):
		start = time.clock()
		dest = cv2.resize(self.img, (IMG_WIDTH, IMG_HEIGHT))
		landmark_rois = self.__get_landmark_rois()
		for slide_window_obj in self.__generat_rect(dest):
			# 迭代结束条件
			need_find_roi = [landmark_roi for landmark_roi in landmark_rois if landmark_roi.times == 0]
			if len(need_find_roi) == 0:
				# print("need find roi is {}".format(len(need_find_roi)))
				break

			for landmark_roi in landmark_rois:
				task = gevent.spawn(self.__final_recognize, landmark_roi, slide_window_obj)
				task.join()

		# position_dic = {}
		real_positions = self.__landmark_position_dic()
		for landmark_roi in landmark_rois:
			landmark = landmark_roi.landmark
			if landmark is None:
				print("{} miss landmark".format(landmark_roi.label))
				continue
			print("############################{}################################################".format(
				landmark_roi.label))

			col = landmark.col
			row = landmark.row

			# print(col, row)
			# print("#" * 100)
			# print(real_positions)
			cv2.rectangle(dest, (col, row), (col + landmark.width, row + landmark.height), color=(0, 255, 255),
			              thickness=2)

			cv2.putText(dest,
			            "{}".format(landmark_roi.label),
			            (col, row + 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
		position_dic = self.choose_best_cornors()
		# print("#" * 100)
		# print(position_dic)
		# dest = self.__perspective_transform(dest, position_dic)

		self.__draw_grid_lines(dest)
		end = time.clock()
		print("结束{}".format(end - start))
		return dest

	def __get_landmark_rois(self):
		landmark_rois = [LandMarkRoi(img=cv2.imread(os.path.join(ROIS_DIR, roi_img)), label=roi_img.split('.')[0], id=1)
		                 for
		                 roi_img in
		                 os.listdir(ROIS_DIR)]
		return landmark_rois

	def __draw_grid_lines(self, img):
		H_rows, W_cols = img.shape[:2]
		for row in range(0, H_rows):
			if row % 50 == 0:
				cv2.line(img, (0, row), (W_cols, row), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)
		for col in range(0, W_cols):
			if col % 50 == 0:
				cv2.line(img, (col, 0), (col, H_rows), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8)

	def __perspective_transform(self, src, position_dic):
		'''透视变化'''
		H_rows, W_cols = src.shape[:2]
		# print(H_rows, W_cols)
		detected_landmarks = len(position_dic.items())

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}
		print(real_position_dic)

		if position_dic is not None and detected_landmarks >= 3:
			left_points = [(landmark_name, (x, y)) for landmark_name, (x, y) in position_dic.items() if
			               x < 0.5 * W_cols]
			right_points = list(filter(lambda item: item[1][0] > 0.5 * W_cols, position_dic.items()))
			left_points = sorted(left_points, key=cmp_to_key(landmarkname_cmp), reverse=False)
			right_points = sorted(right_points, key=cmp_to_key(landmarkname_cmp), reverse=False)
		else:
			# 处理失败直接返回
			mylog_error("检测到的地标小于三个，无法使用")
			return src

		left_points = []
		right_points = []
		for label, [x, y] in position_dic.items():
			if "L" in label:
				left_points.append((label, [x, y]))
			else:
				right_points.append((label, [x, y]))
		left_points.sort(key=lambda point: point[1][1])
		right_points.sort(key=lambda point: point[1][1])

		p1 = left_points[0][1]
		p2 = right_points[0][1]
		p3 = left_points[1][1]
		p4 = right_points[1][1]

		pts1 = np.float32([p1, p3, p2, p4])
		pts2 = np.float32([real_position_dic.get(left_points[0][0]), real_position_dic.get(left_points[1][0]),
		                   real_position_dic.get(right_points[0][0]), real_position_dic.get(right_points[1][0])])

		# 生成透视变换矩阵；进行透视变换
		M = cv2.getPerspectiveTransform(pts1, pts2)
		dst = cv2.warpPerspective(src, M, (W_cols, H_rows))
		return dst

	def __compare_similar(self, img1, img2):
		if img1 is None or img2 is None:
			return 0
		# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
		# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
		return degree

	def __final_recognize(self, landmark_roi: LandMarkRoi, slide_window_obj: NearLandMark):
		if slide_window_obj is None: return
		col, row, slide_img = slide_window_obj.positioninfo
		roi = cv2.resize(landmark_roi.roi, (SLIDE_WIDTH, SLIDE_HEIGHT))
		similar = self.__compare_similar(roi, slide_img)
		if landmark_roi.label=="NO3_R":
			print("{}  ({},{}) similar is {}".format(landmark_roi.label, col, row, similar))
		global step, fail_time, ALL_LANDMARKS_DICT
		if similar >= 0.5:
			slide_window_obj.similarity = similar
			slide_window_obj.land_name = landmark_roi.label

			landmark_roi.set_match_obj(slide_window_obj)

			ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj
			fail_time = 0
			good_rects.append(TargetRect((col - FOND_RECT_WIDTH,
			                              row - FOND_RECT_HEIGHT),
			                             (col + FOND_RECT_WIDTH,
			                              row + FOND_RECT_HEIGHT)))
		# return slide_window_obj
		elif landmark_roi.label in slide_window_obj.maybe_labels and landmark_roi.label not in ALL_LANDMARKS_DICT \
				and similar < 0.5:
			slide_window_obj.similarity = similar
			slide_window_obj.land_name = landmark_roi.label
			landmark_roi.set_match_obj(slide_window_obj)
			ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj

	def __landmark_position_dic(self):
		'''获取所有的地标标定位置'''
		with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
			real_positions = pickle.load(coordinate)
		return real_positions

	def __generat_rect(self, dest=None):
		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 100<=cv2.contourArea(c) < 300
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 10 < rect_w <= 30 and 10 < rect_h <= 30

		global rows, cols, step
		landmark_rois = self.__get_landmark_rois()

		target = cv2.resize(dest, (IMG_WIDTH, IMG_HEIGHT))
		target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
		cv2.imshow("target_binary", target)

		# 思路：通过阈值化缩小比对图像区域
		for roi_template in landmark_rois:
			img_roi_hsvt = cv2.cvtColor(roi_template.roi, cv2.COLOR_BGR2HSV)
			# img_roi_hsvt = roi_template.roi
			roihist = cv2.calcHist([img_roi_hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

			cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
			bk = cv2.calcBackProject([target_hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
			#
			# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			# backproject = cv2.filter2D(backproject, -1, disc)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

			thresh=cv2.dilate(bk,kernel)
			# if roi_template.label=="NO3_R":
			# 	cv2.imshow("bk", thresh)
			ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
			# thresh=cv2.fastNlMeansDenoisingMulti(thresh,2,5,None,4,7,35)

			# 使用merge变成通道图像
			# thresh = cv2.merge((thresh, thresh, thresh))

			# thresh = cv2.medianBlur(thresh, 3)
			# thresh=cv2.bilateralFilter(thresh,d=0,sigmaColor=90,sigmaSpace=7)
			if roi_template.label == 'NO3_R':
				cv2.imshow("thresh", thresh)


			contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			print("{} contours size {}".format(roi_template.label, len(contours)))
			# cv2.drawContours(target, contours, -1, (0, 255, 255), 3)
			if contours is None or len(contours) == 0 :
				continue

			# Z轴无论再怎么变化，灯的面积也大于100
			contours = filter(lambda c: warp_filter(c), contours)
			for c in contours:
				rect = cv2.boundingRect(c)
				rect_x, rect_y, rect_w, rect_h = rect
				# if roi_template.label == 'NO3_R':
				# 	print("NO3_R contour area {}".format(cv2.contourArea(c)))
				# cv2.rectangle(target, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color=(0, 255, 0),
				#               thickness=3)

				landmark_obj = NearLandMark(rect_x, rect_y, dest[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w])
				landmark_obj.width = rect_w
				landmark_obj.height = rect_h
				landmark_obj.add_maybe_label(roi_template.label)
				self.landmarks.append(landmark_obj)
				yield landmark_obj



if __name__ == '__main__':
	# 218  240  60
	src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-15-15-59-16test.bmp')).position_landmark()
	# src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-14-12-50-58test.bmp')).position_landmark()
	b = BagDetector(src)
	print(b.location_bag())

	# __draw_grid_lines(src)
	cv2.namedWindow("dest")
	cv2.imshow("dest", src)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

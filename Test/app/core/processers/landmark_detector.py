# -*- coding: utf-8 -*-
# from gevent import monkey;
import math
import os
import pickle
import random
from collections import defaultdict
from functools import cmp_to_key
import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, ROIS_DIR, LEFT_MARK_FROM, LEFT_MARK_TO, RIGHT_MARK_FROM, RIGHT_MARK_TO, \
	PROGRAM_DATA_DIR
import cv2
import time
import gevent
import profile
from app.core.beans.models import LandMarkRoi, NearLandMark, TargetRect
from app.core.exceptions.allexception import NotFoundLandMarkException
from app.core.processers.bag_detector import BagDetector
from app.core.processers.preprocess import AbstractDetector
from app.log.logtool import mylog_error
import re

cv2.useOptimized()

rows, cols = IMG_HEIGHT, IMG_WIDTH

WITH_TRANSPORT = True


class LandMarkDetecotr(AbstractDetector):
	landmark_match = re.compile(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''')

	def __init__(self):
		self.img_after_modify = None
		self._rois = []
		self.ALL_LANDMARKS_DICT = {}
		self.ALL_POSITIONS = {}

	def landmarkname_cmp(self, a, b):
		result_a = re.match(self.landmark_match, a[0])

		result_b = re.match(self.landmark_match, b[0])

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

	def corners_levelfour(self, left_top_landmark_name):
		'''级别4获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=1)]

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}

		HAS_POINT_NOT_EXIST = False
		for label_item in labels:
			if label_item not in find:
				HAS_POINT_NOT_EXIST = True
				break
		else:
			HAS_POINT_NOT_EXIST = False

		if not HAS_POINT_NOT_EXIST:
			return labels
		else:
			return None

	def corners_levelsix(self, left_top_landmark_name):
		'''级别6获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=0),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=2),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=0, west_step=2)]

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}

		HAS_POINT_NOT_EXIST = False
		for label_item in labels:
			if label_item not in find:
				HAS_POINT_NOT_EXIST = True
				break
		else:
			HAS_POINT_NOT_EXIST = False

		if not HAS_POINT_NOT_EXIST:
			return labels
		else:
			return None

	def corners_leveleight(self, left_top_landmark_name):
		'''级别8获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=3),
		          self.__fetch_neigbbour(left_top_landmark_name, west_step=3)]
		return labels

	def __fetch_neigbbour(self, landmark_name, sourth_step: int = 0, west_step: int = 0):
		result = re.match(self.landmark_match, landmark_name)
		if result is None:
			return landmark_name
		current_no = int(result.group(1))
		current_direct = result.group(2)
		# next_direct = "R" if current_direct == "L" else "L"
		direct = "R" if sourth_step == 1 else current_direct

		no = current_no + west_step if west_step > 0 else current_no
		landmark_labelname = "NO{NO}_{D}".format(NO=no, D=direct)
		return landmark_labelname

	def get_next_no(self, landmark_name, forward=False):
		result = re.match(self.landmark_match, landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))

		next_no = current_no + 1 if not forward else current_no - 1
		next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
		return next_landmark

	def get_opposite_landmark(self, landmark_name):
		import re
		result = re.match(self.landmark_match, landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))
		current_d = result.group(2)
		opposite_d = 'R' if current_d == 'L' else 'L'
		next_landmark = "NO{NO}_{D}".format(NO=current_no, D=opposite_d)
		return next_landmark

	def compute_miss_landmark_position(self, landmark_name):
		opposite = self.get_opposite_landmark(landmark_name)
		print("opposite is {}".format(opposite))
		if opposite not in self.ALL_LANDMARKS_DICT:
			raise NotFoundLandMarkException("opposite landmark is not exist")
		# assert opposite in ALL_LANDMARKS_DICT, "opposite landmark is not exist"

		y = self.ALL_LANDMARKS_DICT[opposite].row

		rows = int(len(self.rois) / 2)
		x = 0
		index = -rows
		while index < rows:
			if index < 0:
				label = self.get_next_no(landmark_name, forward=True)
			else:
				label = self.get_next_no(landmark_name, forward=False)
			try:
				x = self.ALL_LANDMARKS_DICT[label].col
				break
			except:
				index += 1
				continue

		return x, y

	def choose_best_cornors(self):
		'''
		 method choose_best_cornors :must have three landmarks in your image,if not ,then send move instruct to computer
		  <1> check  if  has three point in image,if miss one ,the method may compute it by others landmark;if less than 3,then return back....
		  <2> send move instruct is not in use
		:return: positiondict:{} ,success:boolean
		'''

		find = {roi_item.label: 1 if roi_item.label in self.ALL_LANDMARKS_DICT else 0 for roi_item in self.rois}
		landmark_total = sum(find.values())
		if landmark_total < 3:
			return {}, False

		level_four_min_loss, level_six_min_loss, best_four_label_choose, best_six_label_choose = 4, 6, None, None

		best_four_label_choose, best_six_label_choose, loss = self.position_loss(best_four_label_choose,
		                                                                         best_six_label_choose, find,
		                                                                         level_four_min_loss,
		                                                                         level_six_min_loss)
		# 记录每个地标的定位目标
		positiondict = {}
		if best_four_label_choose not in find and best_six_label_choose not in find:
			# 评级4与评级5的坐标点 均无效
			return {}, False
		elif best_four_label_choose in find and best_six_label_choose in find:
			# 评级4与评级5的坐标点 均有效
			labels = self.corners_levelfour(best_four_label_choose) if loss[best_four_label_choose]['4'] < \
			                                                           loss[best_six_label_choose][
				                                                           '6'] else self.corners_levelsix(
				best_six_label_choose)
		elif (best_four_label_choose not in find and best_six_label_choose in find) or (
				best_four_label_choose in find and best_six_label_choose not in find):
			# 评级4与评级5的坐标点 其中一个有效
			labels = self.corners_levelfour(
				best_four_label_choose) if best_four_label_choose in find else self.corners_levelsix(
				best_six_label_choose)
		# 并计算丢失的地标，如果需要四个地标只有三个获取，缺失的需要计算推导出来
		compensate_label = ""
		for label in labels:
			if label not in self.ALL_LANDMARKS_DICT:
				print("label {} need compute".format(label))
				compensate_label = label
				continue
			else:
				positiondict[label] = [self.ALL_LANDMARKS_DICT[label].col, self.ALL_LANDMARKS_DICT[label].row]

		if compensate_label != "":
			try:
				miss_x, miss_y = self.compute_miss_landmark_position(compensate_label)
			except (NotFoundLandMarkException) as e:
				return {}, False
			else:
				positiondict[compensate_label] = [miss_x, miss_y]
		############ -----------------开始间隔检测-------------------------------------------------------------#########
		success = True

		for key, [x, y] in positiondict.items():
			key_result = re.match(self.landmark_match, key)
			key_no = key_result.group('NO')
			key_direct = key_result.group('direct')
			for key_j, [xj, yj] in positiondict.items():
				keyj_result = re.match(self.landmark_match, key)
				keyj_no = keyj_result.group('NO')
				keyj_direct = keyj_result.group('direct')
				if key == key_j:
					continue
				if (key_no < keyj_no and y > yj) or (key_no > keyj_no and y < yj):
					return positiondict, False
				if key_j != key and abs(x - xj) < 20 and abs(y - yj) < 20:
					return positiondict, False
				if key_direct == keyj_direct:
					q = math.sqrt(math.pow(abs(xj - x), 2) + math.pow(abs(yj - y), 2))
					if q < 100:
						return {}, False
		##########------------------结束间隔检测-------------------------------------------------------------#########
		###########-------------- 开始递增顺序检测------------------------------------------------------------#########
		position_row_table = defaultdict(list)

		for label, [x, y] in positiondict.items():
			item_match_result = re.match(self.landmark_match, label)
			position_row_table[item_match_result.group('NO')].append(y)

		position_row_table = {item[0]: item[1] for item in
		                      sorted(position_row_table.items(), key=lambda record: record[0], reverse=False)}
		position_row_temp = {}
		for label, row_list in positiondict.items():
			item_match_result = re.match(self.landmark_match, label)
			average = sum(row_list) / len(row_list)
			position_row_temp[item_match_result.group('NO')] = average
			priver_no = str(int(item_match_result.group('NO')) - 1)
			# 不按照编号递增排序
			if priver_no in position_row_temp and (
					position_row_temp[priver_no] >= average or abs(position_row_temp[priver_no] - average) < 190):
				break
			# 差异太大应该放弃
			score = sum([math.pow(row - average, 2) for row in row_list])
			if score > 20:
				break
		else:
			# --------------------------------判断是否错位--------------------------------------------------------
			left_points, right_points = [], []
			for label, point in positiondict.items():
				if '_L' in label:
					left_points.append(point[y])
				else:
					right_points.append(point[y])
			# 结束--------------------y轴扭曲--------------------------------------------
			left_points = sorted(left_points, lambda p: p[1], reverse=False)
			right_points = sorted(right_points, lambda p: p[1], reverse=False)
			top_range = abs(left_points[0] - right_points[0])
			bottom_range = abs(left_points[1] - right_points[1])
			if abs(top_range - bottom_range) > 30:
				return positiondict, False
		###################-----------------结束递增顺序检测-------------------------------#####################
		return positiondict, success

	def position_loss(self, best_four_label_choose, best_six_label_choose, find, level_four_min_loss,
	                  level_six_min_loss):
		'''
		计算丢失值，用来量化
		:param best_four_label_choose:
		:param best_six_label_choose:
		:param find:
		:param level_four_min_loss:
		:param level_six_min_loss:
		:return:
		'''
		loss = {}  # 闭合层级：丢失角点个数
		# 计算每个地标的定位识别状态
		for roi_item in self.rois:
			if "_R" in roi_item.label:
				continue
			loss_info = {}
			for level in ['4', '6']:
				label = roi_item.label
				if level == '4':
					candidate_landmarks = self.corners_levelfour(label)
					if candidate_landmarks is None:
						continue
					else:
						point1, point2, point3, point4 = candidate_landmarks

						loss_info['4'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
						if loss_info['4'] < level_four_min_loss:
							best_four_label_choose = roi_item.label
							level_four_min_loss = loss_info['4']

				elif level == '6':
					candidate_landmarks = self.corners_levelsix(label)
					if candidate_landmarks is None:
						continue
					else:
						point1, point2, point3, point4 = candidate_landmarks
						loss_info['6'] = 4 - sum([find[point1], find[point2], find[point3], find[point4]])
						if loss_info['6'] < level_six_min_loss:
							best_six_label_choose = roi_item.label
							level_six_min_loss = loss_info['6']

				loss[label] = loss_info
		return best_four_label_choose, best_six_label_choose, loss

	def position_landmark(self, image):
		self.ALL_LANDMARKS_DICT.clear()
		self.ALL_POSITIONS.clear()
		del self.rois
		start = time.perf_counter()
		rows, cols, channels = image.shape
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			dest = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
		else:
			dest = image

		self.candidate_landmarks(dest)
		for label, (x, y) in self.ALL_POSITIONS.items():
			label_match_result = re.match(self.landmark_match, label)
			no = label_match_result.group('NO')
			exception_labels = [landmarkname for landmarkname, (itemx, itemy) in self.ALL_POSITIONS.items() if
			                    (re.match(self.landmark_match, landmarkname).group('NO') < no and itemy > y) or (
						                    re.match(self.landmark_match, landmarkname).group('NO') > no and itemy < y)]
			if len(exception_labels) > 0:
				for exception_label in exception_labels:
					print("{}is {}，but {} is {}".format(label, y, exception_label,
					                                    self.ALL_POSITIONS[exception_label][1]))
				return dest, False

		if len(self.ALL_LANDMARKS_DICT.keys()) < 3:
			return dest, False

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}

		for landmark_roi in self.rois:
			landmark = landmark_roi.landmark
			if landmark is None or landmark_roi.label not in self.ALL_LANDMARKS_DICT:
				self.logger("{} miss landmark".format(landmark_roi.label), "warn")
				continue

			col = landmark.col
			row = landmark.row
			real_col, real_row = real_position_dic[landmark_roi.label]
			cv2.rectangle(dest, (col, row), (col + landmark.width, row + landmark.height), color=(255, 0, 255),
			              thickness=2)
			cv2.putText(dest,
			            "({},{})".format(real_col, real_row),
			            (col, row + 90),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
			cv2.putText(dest,
			            "{}".format(landmark_roi.label),
			            (col, row + 60),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		# 获取最佳的四个地标，如果缺失一个可以通过计算获取
		position_dic, success = self.choose_best_cornors()
		if success:
			dest, success = self.__perspective_transform(dest, position_dic)
			if success: self.draw_grid_lines(dest)
		end = time.perf_counter()
		print("结束{}".format(end - start))
		return dest, success

	@property
	def rois(self):
		if self._rois is None or len(self._rois) == 0:
			landmark_rois = [
				LandMarkRoi(img=cv2.imread(os.path.join(ROIS_DIR, roi_img)), label=roi_img.split('.')[0], id=1)
				for
				roi_img in
				os.listdir(ROIS_DIR)]
			self._rois = landmark_rois
		return self._rois

	@rois.deleter
	def rois(self):
		if self._rois is not None:
			self._rois.clear()

	def draw_grid_lines(self, img):
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

		if position_dic is not None and detected_landmarks >= 3:
			left_points = [(landmark_name, (x, y)) for landmark_name, (x, y) in position_dic.items() if
			               x < 0.5 * W_cols]
			right_points = list(filter(lambda item: item[1][0] > 0.5 * W_cols, position_dic.items()))
			left_points = sorted(left_points, key=cmp_to_key(self.landmarkname_cmp), reverse=False)
			right_points = sorted(right_points, key=cmp_to_key(self.landmarkname_cmp), reverse=False)
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

		try:
			p1 = left_points[0][1]
			p2 = right_points[0][1]
			p3 = left_points[1][1]
			p4 = right_points[1][1]
		except:
			return src, False
		else:
			pts1 = np.float32([p1, p3, p2, p4])
			pts2 = np.float32([real_position_dic.get(left_points[0][0]), real_position_dic.get(left_points[1][0]),
			                   real_position_dic.get(right_points[0][0]), real_position_dic.get(right_points[1][0])])

			# 生成透视变换矩阵；进行透视变换
			M = cv2.getPerspectiveTransform(pts1, pts2)
			dst = cv2.warpPerspective(src, M, (W_cols, H_rows))
		return dst, True

	def __compare_hsv_similar(self, img1, img2):
		if img1 is None or img2 is None:
			return 0
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA    HISTCMP_CORREL
		return degree

	def __compare_rgb_similar(self, img1, img2):
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
		roi = cv2.resize(landmark_roi.roi, (slide_window_obj.width, slide_window_obj.height))
		similar_rgb = self.__compare_rgb_similar(roi, slide_img)
		hsv_similar = self.__compare_hsv_similar(roi, slide_img)
		# if landmark_roi.label=="NO1_L":
		# 	print("{}  ({},{}) similar is {}".format(landmark_roi.label, col, row, similar))
		slide_window_obj.similarity = max(similar_rgb, hsv_similar)
		slide_window_obj.land_name = landmark_roi.label
		if similar_rgb >= 0.5 or hsv_similar > 0.5:

			landmark_roi.set_match_obj(slide_window_obj)
			self.ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj
			fail_time = 0
		else:
			for label, exist_land in self.ALL_LANDMARKS_DICT.items():
				if exist_land.col == slide_window_obj.col and exist_land.row == slide_window_obj.row: break
			else:
				if landmark_roi.landmark is None:
					# now  i have try my best to decide if  put in
					# may be another feature may work
					landmark_roi.set_match_obj(slide_window_obj)
					self.ALL_LANDMARKS_DICT[landmark_roi.label] = slide_window_obj

	def __landmark_position_dic(self):
		'''获取所有的地标标定位置'''
		with open(os.path.join(PROGRAM_DATA_DIR, 'coordinate_data.txt'), 'rb') as coordinate:
			real_positions = pickle.load(coordinate)
		return real_positions

	def candidate_landmarks(self, dest=None, left_start=210, left_end=260, right_start=670, right_end=750):
		'''dest=None, left_start=230, left_end=300, right_start=500, right_end=750'''
		global rows, cols, step

		# 不要忽略缩小图片尺寸的重要性，减小尺寸，较少像素数就可以最大限度的减少无用操作；
		# 限制程序速度的最主要因素就是无用操作，无用操作越少，程序执行速度就越高。
		target = cv2.resize(dest, (IMG_WIDTH, IMG_HEIGHT))
		# HSV对光线较RGB有更好的抗干扰能力
		target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
		# cv2.imshow("target", target)
		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		img_world = np.ones_like(gray)
		ret, img_world = cv2.threshold(img_world, 0, 255, cv2.THRESH_BINARY)

		# cv2.imshow("first", img_world)

		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 200 <= cv2.contourArea(c) < 3600
			rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig and 3 < rect_w <= 60 and 3 < rect_h <= 60

		def set_mask_area(x: int, y: int, width: int, height: int):
			img_world[y:y + height, x:x + width] = 0

		left_open_mask = np.zeros_like(gray)
		left_open_mask[0:IMG_HEIGHT, left_start:left_end] = 255

		right_open_mask = np.zeros_like(gray)
		right_open_mask[0:IMG_HEIGHT, right_start:right_end] = 255

		bigest_h, bigest_w = 0, 0

		for roi_template in self.rois:
			img_roi_hsvt = cv2.cvtColor(roi_template.roi, cv2.COLOR_BGR2HSV)
			# cv2.imshow("roihist",img_roi_hsvt)
			img_roi_hsvt = img_roi_hsvt
			# roihist = cv2.calcHist([img_roi_hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
			#
			# cv2.normalize(roihist, roihist, 0, 256, cv2.NORM_MINMAX)
			# foreground = cv2.calcBackProject([target_hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
			foreground = self.find_it(target_hsvt, img_roi_hsvt)

			# 用来测试
			# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
			# foreground = cv2.filter2D(foreground, -1, disc)

			# foreground = cv2.resize(foreground, (IMG_WIDTH, IMG_HEIGHT))

			if roi_template.label.find("L") > 0:
				foreground = cv2.bitwise_and(foreground, foreground, mask=left_open_mask)
			if roi_template.label.find("R") > 0:
				foreground = cv2.bitwise_and(foreground, foreground, mask=right_open_mask)

			foreground = cv2.bitwise_and(foreground, foreground, mask=img_world)

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			foreground = cv2.dilate(foreground, kernel)
			# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			# foreground = cv2.filter2D(foreground, -1, disc)
			ret, foreground = cv2.threshold(foreground, 110, 255, cv2.THRESH_BINARY)

			# if roi_template.label == "NO1_R":
			# 	cv2.imshow("NO1_R", foreground)
			# cv2.imshow("NO1_R", foreground)
			# cv2.imshow("imgworld", img_world)

			# thresh=cv2.fastNlMeansDenoisingMulti(thresh,2,5,None,4,7,35)

			# 使用merge变成通道图像
			# thresh = cv2.merge((thresh, thresh, thresh))

			# bk = cv2.medianBlur(bk, 3)
			# thresh=cv2.bilateralFilter(thresh,d=0,sigmaColor=90,sigmaSpace=7)
			# if roi_template.label == 'NO1_R':
			# 	cv2.imshow("missed_landmark", bk)

			contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			# if contours is None or len(contours)==0:
			# 	# 前景图有白影 但是检测不到轮廓
			# 	pass
			# if roi_template.label == "NO2_R":
			# 	cv2.imshow("NO2_R", foreground)
			# 	# cv2.imshow("NO1_R", foreground)
			# 	cv2.imshow("imgworld", img_world)
			# 	print("contours is {}".format(len(contours)))

			contours = list(filter(lambda c: warp_filter(c), contours)) if len(contours) > 1 else contours

			if contours is None or len(contours) == 0:
				continue

			# Z轴无论再怎么变化，灯的面积也大于90

			max_area = 0
			best_match_contour = None

			for c in contours:
				area = cv2.contourArea(c)
				if area > max_area:
					max_area = area
					best_match_contour = c

				M = cv2.moments(c)
				try:
					center_x = int(M["m10"] / M["m00"])
					center_y = int(M["m01"] / M["m00"])
				except:
					continue
				# print(roi_template.label,center_x,center_y)

				rect = cv2.boundingRect(c)
				x, y, w, h = rect
				if h > bigest_h: bigest_h = h
				if w > bigest_w: bigest_w = w
				neighbours = [('for_row', self.get_opposite_landmark(roi_template.label)),
				              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=1)),
				              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=2)),
				              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=-1)),
				              ('for_col', self.__fetch_neigbbour(roi_template.label, sourth_step=0, west_step=-2))]
				for flag, ref_label in neighbours:
					if flag == 'for_col' and ref_label in self.ALL_LANDMARKS_DICT:
						ref_landmark = self.ALL_LANDMARKS_DICT[ref_label]
						if abs(ref_landmark.col - x) <= 50:
							landmark_obj = NearLandMark(x, y,
							                            target[y:y + h, x:x + w])
							landmark_obj.width = max(bigest_w, w)
							landmark_obj.height = max(bigest_h, h)
							set_mask_area(center_x - 50, center_y - 50, 200, 200)
							landmark_obj.add_maybe_label(roi_template.label)
							roi_template.set_match_obj(landmark_obj)
							self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj
							break
					elif flag == 'for_row' and ref_label in self.ALL_LANDMARKS_DICT:
						ref_landmark = self.ALL_LANDMARKS_DICT[ref_label]
						if abs(ref_landmark.row - y) <= 50:
							landmark_obj = NearLandMark(x, y,
							                            target[y:y + h, x:x + w])
							landmark_obj.width = max(bigest_w, w)
							landmark_obj.height = max(bigest_h, h)
							set_mask_area(center_x - 50, center_y - 50, 200, 200)
							landmark_obj.add_maybe_label(roi_template.label)
							roi_template.set_match_obj(landmark_obj)
							self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj
							break
				else:
					# for _else   not if else
					# 这类最难处理，处理一对多的情况
					rect = cv2.boundingRect(best_match_contour)
					best_x, best_y, best_w, best_h = rect
					landmark_obj = NearLandMark(best_x, best_y,
					                            target[best_y:best_y + best_h, best_x:best_x + best_w])
					landmark_obj.width = max(bigest_w, best_w)
					landmark_obj.height = max(bigest_h, best_h)
					landmark_obj.add_maybe_label(roi_template.label)
					if len(contours) == 1:
						set_mask_area(center_x - 50, center_y - 50, 200, 200)
					roi_template.set_match_obj(landmark_obj)
					self.ALL_LANDMARKS_DICT[roi_template.label] = landmark_obj

		# print(self.ALL_LANDMARKS_DICT)
		need_delete_keys = []
		for key, landmarkitem in self.ALL_LANDMARKS_DICT.items():
			self.ALL_POSITIONS[key] = (landmarkitem.col, landmarkitem.row)
			east_n = self.__fetch_neigbbour(key, west_step=-1)
			west_n = self.__fetch_neigbbour(key, west_step=1)
			if east_n in self.ALL_LANDMARKS_DICT:
				east_landmark = self.ALL_LANDMARKS_DICT[east_n]
				if east_landmark.row > landmarkitem.row:
					need_delete_keys.append(key)

			if west_n in self.ALL_LANDMARKS_DICT:
				west_landmark = self.ALL_LANDMARKS_DICT[west_n]
				if west_landmark.row < landmarkitem.row:
					need_delete_keys.append(key)
		for key in need_delete_keys:
			del self.ALL_LANDMARKS_DICT[key]
			del self.ALL_POSITIONS[key]


def test_one_image():
	a = LandMarkDetecotr()
	# cap = cv2.VideoCapture("C:/NTY_IMG_PROCESS/VIDEO/Video_20200519095438779.avi")
	# cap.set(cv2.CAP_PROP_POS_FRAMES, 243)
	# ret, frame = cap.read()
	# frame=cv2.resize(frame,(IMG_HEIGHT,IMG_WIDTH))
	# cv2.namedWindow("src")
	# cv2.imshow("src",frame)
	# image=cv2.imread("c:/work/nty/hangche/Image_20200522152805570.bmp")
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152729727.bmp")
	# image = cv2.imread("c:/work/nty/hangche/Image_20200522152825743.bmp")
	image = cv2.imread("c:/work/nty/hangche/Image_20200522154907736.bmp")
	# image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
	cv2.namedWindow("src")
	cv2.imshow("src", image)
	dest, success = a.position_landmark(image)
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)
	# src = LandMarkDetecotr(img=cv2.imread('d:/2020-05-14-12-50-58test.bmp')).position_landmark()
	b = BagDetector()
	if WITH_TRANSPORT and success:

		for bag in b.location_bags(dest, middle_start=100, middle_end=400):
			print(bag)
		a.draw_grid_lines(dest)
	else:
		for bag in b.location_bags(dest, middle_start=400, middle_end=600):
			print(bag)
	# print(dest.shape)

	# cap.release()
	cv2.namedWindow("dest")
	cv2.imshow("dest", dest)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test_avi():
	video = cv2.VideoCapture("C:/NTY_IMG_PROCESS/VIDEO/Video_20200520142647832.avi")
	video.set(cv2.CAP_PROP_POS_FRAMES, 1)
	# 将视频文件初始化为VideoCapture对象
	success, frame = video.read()
	a = LandMarkDetecotr()
	b = BagDetector()
	stop = 1
	# read()方法读取视频下一帧到frame，当读取不到内容时返回false!
	while success and cv2.waitKey(1) & 0xFF != ord('q'):
		# 等待1毫秒读取键键盘输入，最后一个字节是键盘的ASCII码。ord()返回字母的ASCII码
		# time.sleep(1)
		frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
		dest, success = a.position_landmark(frame)
		if WITH_TRANSPORT and success:
			b.location_bags(dest, success, middle_start=150, middle_end=500)
		cv2.imshow('frame', dest)
		success, frame = video.read()
		# if stop==1:
		# 	cv2.waitKey(0)
		# 	break
		stop += 1
	cv2.destroyAllWindows()
	video.release()


if __name__ == '__main__':
	# test_one_image()
	test_avi()

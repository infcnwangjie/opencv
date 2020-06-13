# -*- coding: utf-8 -*-
import math
import pickle
import cv2
import numpy as np
import re
import ctypes
from ctypes import cdll
from collections import defaultdict
from functools import cmp_to_key, reduce, partial

from app.core.beans.models import *
from app.core.video.mvs.MvCameraSuppl_class import MvSuply
from app.log.logtool import mylog_error, mylog_debug, logger

from app.config import *
from app.core.exceptions.allexception import NotFoundLandMarkException

from app.log.logtool import mylog_error

cv2.useOptimized()
rows, cols = IMG_HEIGHT, IMG_WIDTH
WITH_TRANSPORT = True


# ------------------------------------------------
# 名称：add_picture
# 功能：图片相加
# 参数：[img1] ---
#     [img2] ---
# 返回：array ---
# 作者：王杰  2020-3-20
# ------------------------------------------------
def add_picture(img1, img2):
	return cv2.add(img1, img2)


# ------------------------------------------------
# 名称：sort_bag_contours
# 功能：袋子轮廓按照X轴排序
# 状态：后期要废弃
# 参数：[array] ---
# 返回：array ---
# 作者：王杰  2020-4-15
# ------------------------------------------------
def sort_bag_contours(arr):
	if len(arr) < 2:
		return arr
	elif len(arr) == 2:
		c1, c2 = arr
		c1_x, c1_y, c1_w, c1_h = cv2.boundingRect(c1)
		c2_x, c2_y, c2_w, c2_h = cv2.boundingRect(c2)
		if c1_x > c2_x:
			return [c2, c1]
		else:
			return [c1, c2]

	# print(len(arr)//2)
	mid = arr[len(arr) // 2]
	left, right = [], []
	# arr.remove(mid)

	m_x, m_y, m_w, m_h = cv2.boundingRect(mid)
	for item in arr:
		i_x, i_y, i_w, i_h = cv2.boundingRect(item)
		if i_x >= m_x:
			right.append(item)
		else:
			left.append(item)
	return sort_bag_contours(left) + [mid] + sort_bag_contours(right)


# ------------------------------------------------
# 名称：BaseDetector
# 功能：目标检测基类，提供一些公共的方法
# 状态：在用
# 作者：王杰  注释后补
# ------------------------------------------------
class BaseDetector(object):

	# ------------------------------------------------
	# 名称：logger
	# 功能：封装日志功能
	# 状态：在用
	# 参数： [msg]   ---日志消息
	#        [lever] ---日志级别
	# 返回：None ---
	# 作者：王杰  2020-5-xx
	# ------------------------------------------------
	def logger(self, msg: str, lever='info'):
		from app.log.logtool import logger
		logger(msg, lever)

	# 获取图像行、列总数
	@property
	def shape(self):
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape
		return rows, cols

	# ------------------------------------------------
	# 名称：sharper
	# 功能：凸显边缘信息
	# 状态：在用
	# 参数： [array]   ---输入图像
	# 返回： array --- 输出图像
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def sharper(self, image):
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
		dst = cv2.filter2D(image, -1, kernel=kernel)
		return dst

	# ------------------------------------------------
	# 名称：interpolation_binary_data
	# 功能：插值
	# 状态：在用
	# 参数： [array]   ---输入二进制图像
	# 返回： array --- 输出图像
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def interpolation_binary_data(self, binary_image):
		destimg = np.zeros_like(binary_image)
		cv2.resize(binary_image, destimg, interpolation=cv2.INTER_NEAREST)
		return destimg

	# ------------------------------------------------
	# 名称：color_similar_ratio
	# 功能：图像相似度
	# 状态：在用
	# 参数： [array]   ---目标图像
	#        [array]   ---模板图像
	# 要求： image1与image2 大小一样，至少像素数差不大
	# 返回： float     ---匹配率
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def color_similar_ratio(self, image1, image2):
		if image1 is None or image2 is None:
			return 0
		img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		# cv2.imshow("hist1",hist1)
		hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 255.0])
		cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)  # 规划到0-255之间
		degree = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
		return degree

	# ------------------------------------------------
	# 名称：red_contours
	# 功能：获取红色轮廓
	# 状态：在用
	# 参数： [array]   ---输入图像
	#        [int]    ---开始边界
	#        [int]    ---结束边界
	# 要求： image1与image2 大小一样，至少像素数差不大
	# 返回： float     ---匹配率
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def red_contours(self, img, middle_start=180, middle_end=500):
		# red_low, red_high = [120, 50, 50], [180, 255, 255]
		red_low, red_high = [156, 43, 46], [180, 255, 255]
		red_min, red_max = np.array(red_low), np.array(red_high)
		# 去除颜色范围外的其余颜色
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		red_mask = cv2.inRange(hsv, red_min, red_max)
		ret, red_binary = cv2.threshold(red_mask, 0, 255, cv2.THRESH_BINARY)
		middle_open_mask = np.zeros_like(red_binary)
		middle_open_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255
		red_binary = cv2.bitwise_and(red_binary, red_binary, mask=middle_open_mask)
		red_binary = cv2.medianBlur(red_binary, 3)
		red_contours, _hierarchy = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return red_binary, red_contours

	# ------------------------------------------------
	# 名称：red_contours
	# 功能：获取黄色轮廓
	# 状态：在用
	# 参数： [array]   ---输入图像
	# 要求： img是RGB格式图片
	# 返回： 数组     ---轮廓数组
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def yellow_contours(self, img):
		yellow_low, yellow_high = [11, 43, 46], [34, 255, 255]

		yellow_min, yellow_max = np.array(yellow_low), np.array(yellow_high)
		# 去除颜色范围外的其余颜色

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

		yellow_ret, yellow_binary = cv2.threshold(yellow_mask, 100, 255, cv2.THRESH_BINARY)
		# 去噪
		# yellow_binary = cv2.medianBlur(yellow_binary, 3)

		yellow_contours, _hierarchy = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		return yellow_binary, yellow_contours

	# ------------------------------------------------
	# 名称：green_contours
	# 功能：获取绿色轮廓
	# 状态：在用
	# 参数：  [array]   ---输入图像
	#        [int]   ---开始边界
	#       [int]   ---结束边界
	# 要求： img是RGB格式图片
	# 返回： 数组     ---轮廓数组
	# 作者：王杰  编写 2020-4-xx  修改 2020-6-xx
	# ------------------------------------------------
	def green_contours(self, img, middle_start=100, middle_end=450):
		rows, cols, channels = img.shape
		# 如果尺寸已经调整，就无须调整
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# cv2.imshow("hsv",hsv)
		# green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_low, green_high = [35, 43, 46], [77, 255, 255]
		green_min, green_max = np.array(green_low), np.array(green_high)
		green_mask = cv2.inRange(hsv, green_min, green_max)

		green_ret, foreground = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)

		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		foreground = cv2.filter2D(foreground, -1, disc)

		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# middle_mask = np.zeros_like(gray)
		# middle_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255
		#
		#
		# foreground = cv2.bitwise_and(foreground, foreground, mask=middle_mask)
		# cv2.imshow("green_binary", middle_mask)

		# cv2.imshow("green_binary", foreground)
		# foreground = cv2.medianBlur(foreground, 3)
		green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return foreground, green_contours


# ------------------------------------------------
# 名称：BagDetector
# 功能：袋子检测算法汇总
# 状态：在用
# 作者：王杰  注释后补
# ------------------------------------------------
class BagDetector(BaseDetector):

	def __init__(self, img=None):
		super().__init__()
		self.bags = []

	# ------------------------------------------------
	# 名称：bagroi_templates
	# 功能：获取袋子模型
	# 状态：暂时不用
	# 参数： None   ---
	# 返回： array --- BagRoi object
	# 作者：王杰  2020-4-xx
	# ------------------------------------------------
	def bagroi_templates(self):
		landmark_rois = [BagRoi(img=cv2.imread(os.path.join(BAGROI_DIR, roi_img)), id=index)
		                 for
		                 index, roi_img in
		                 enumerate(os.listdir(BAGROI_DIR)) if roi_img.find('bag') != -1]
		return landmark_rois

	# ------------------------------------------------
	# 名称：findbags
	# 功能：获取袋子模型
	# 状态：暂时不用
	# 参数： [img_copy]       ---输入图像,有个小细节 地标检测、钩子检测、袋子检测提供的输入图片应该都是深拷贝
	#        [middle_start]   ---开始边界
	#        [middle_end]     ---结束边界
	# 返回： array --- BagContour object
	# 作者：王杰     编写 ： 2020-3-xx    修改 ： 2020-6-xx
	# ------------------------------------------------
	def findbags(self, img_copy=None, middle_start=300, middle_end=500):
		def warp_filter(c):
			'''内部过滤轮廓'''
			area = cv2.contourArea(c)
			isbig = (area > 30 and area < 10000)
			# rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			return isbig

		global rows, cols, step
		# target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

		# gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
		cols, rows, channels = img_copy.shape
		# print(rows,cols,channels)
		foreground, contours = self.red_contours(img_copy, middle_start, middle_end)

		ret, foreground = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Z轴无论再怎么变化，灯的面积也大于90
		if contours is not None and len(contours) > 0:
			contours = list(filter(lambda c: warp_filter(c), contours))
			logger("bag contour is {}".format(len(contours)), level='info')

		return [BagContour(c) for c in contours], contours, foreground

	# ------------------------------------------------
	# 名称：location_bags_withlandmark
	# 功能：用地标定位袋子，另外还有一个方法location_bags_withoutlandmark.
	#       有这么一个细节，位置检测与落钩选择不同定位方法
	# 状态：在用
	# 参数：  [dest]            ---用于画轮廓
	#        [img_copy]         ---为了排除钩子检测、地标检测，分配的图像空间，深拷贝
	#       [success_location]  ---地标定位成功与否
	#       [middle_start]      ---开始边界
	#       [middle_end]        ---结束边界
	# 返回： [数组]     ---Bag Array
	#        [bagforeground]     ---检测袋子前景图像，为方便测试，留了一个接口
	# 作者：王杰  编写 2020-4-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_bags_withlandmark(self, dest, img_copy, success_location=True, middle_start=110, middle_end=500):

		# self.bags.clear()
		# 存储前景图
		bagcontours, contours, foreground = self.findbags(img_copy, middle_start, middle_end)
		cv2.drawContours(dest, contours, -1, (170, 0, 255), 3)

		# 对袋子排序
		orderby_x_bagcontours = sorted(bagcontours, key=lambda bagcontour: cv2.boundingRect(bagcontour.c)[0],
		                               reverse=False)
		for x_index, orderby_x_c in enumerate(orderby_x_bagcontours):
			orderby_x_c.col_sort_index = x_index + 1

		orderby_y_bagcontours = sorted(orderby_x_bagcontours, key=lambda bagcontour: cv2.boundingRect(bagcontour.c)[1],
		                               reverse=False)
		for y_index, orderby_y_c in enumerate(orderby_y_bagcontours):
			orderby_y_c.row_sort_index = y_index + 1

		for bag_contour in orderby_y_bagcontours:
			bag = Bag(bag_contour)
			bag.modify_box_content()
			if success_location:
				cv2.putText(dest, bag.box_content,
				            (bag.boxcenterpoint[0], bag.boxcenterpoint[1] + 10),
				            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

			# 只检测移动状态未结束的袋子
			for existbag in [item for item in self.bags if item.status_map['finish_move'] == False]:
				if abs(existbag.x - bag.x) < 20 and abs(existbag.y - bag.y) < 20:
					existbag.x, existbag.y = bag.x, bag.y
					if bag.img_x is not None and bag.img_y is not None:
						existbag.img_x, existbag.img_y = bag.img_x, bag.img_y
					break
			else:
				self.bags.append(bag)

		return self.bags, foreground

	# ------------------------------------------------
	# 名称：location_bags_withoutlandmark
	# 功能：用地标定位袋子，另外还有一个方法location_bags_withlandmark.
	#       有这么一个细节，位置检测与落钩选择不同定位方法
	# 状态：在用
	# 参数：  [original_img]            ---用于画轮廓
	#       [middle_start]      ---开始边界
	#       [middle_end]        ---结束边界
	# 返回： [数组]     ---Bag Array
	#        [bagforeground]     ---检测袋子前景图像，为方便测试，留了一个接口
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_bags_withoutlandmark(self, original_img, middle_start=245, middle_end=490):
		'''
		 cm:def location_bags(self,target,success_location=True,middle_start=150,middle_end=500):
		'''
		# self.bags.clear()
		# 存储前景图
		bagcontours, contours, foreground = self.findbags(original_img, middle_start, middle_end)
		# cv2.drawContours(original_img, contours, -1, (170, 0, 255), 3)

		# 先按照袋子X轴进行排序
		orderby_x_bagcontours = sorted(bagcontours, key=lambda bagcontour: cv2.boundingRect(bagcontour.c)[0],
		                               reverse=False)
		for x_index, orderby_x_c in enumerate(orderby_x_bagcontours):
			orderby_x_c.col_sort_index = x_index + 1

		# 先按照袋子Y轴进行排序
		orderby_y_bagcontours = sorted(orderby_x_bagcontours, key=lambda bagcontour: cv2.boundingRect(bagcontour.c)[1],
		                               reverse=False)
		for y_index, orderby_y_c in enumerate(orderby_y_bagcontours):
			orderby_y_c.row_sort_index = y_index + 1

		bags_withoutlandmark = []
		for bag_contour in orderby_y_bagcontours:
			bag = Bag(bag_contour)
			bag.modify_box_content()

			# 建议保留，标注袋子坐标
			cv2.putText(original_img, bag.box_content, (bag.boxcenterpoint[0], bag.boxcenterpoint[1] + 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

			bags_withoutlandmark.append(bag)

		return bags_withoutlandmark, foreground


# ------------------------------------------------
# 名称：LandMarkDetecotr
# 功能：地标检测算法.
# 状态：在用
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class LandMarkDetecotr(BaseDetector):
	'''
	地标检测算法
	'''
	landmark_match = re.compile(r'''NO(?P<NO>[0-9]*)_(?P<direct>[A-Z]{1})''')

	def __init__(self):
		self.img_after_modify = None
		self._rois = []
		self.ALL_LANDMARKS_DICT = {}
		self.ALL_POSITIONS = {}

	# ------------------------------------------------
	# 名称： landmarkname_cmp
	# 功能： 检测地标编号大小
	# 状态：在用
	# 参数：  [a]     ---例如地标 NO2-L
	#        [b]      ---例如地标 NO3-R
	# 返回： [int]    ---NO2-L与NO3-R返回-1
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称： corners_levelfour
	# 功能： 检测四个角点
	# 初衷： 坐标系标定需要四个角点
	# 状态： 在用
	# 参数： [left_top_landmark_name]     ---例如地标 NO2-L
	# 返回： [array]    ---NO2-L NO2-R NO3-L NO3-R
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
	def corners_levelfour(self, left_top_landmark_name):
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

	# ------------------------------------------------
	# 名称： corners_levelsix
	# 功能： 检测四个角点 ，不是NO1-L NO2-L NO1-R NO1-R，而是NO1-L NO3-L NO1-R NO3-R
	# 初衷： 坐标系标定需要四个角点
	# 状态： 在用
	# 参数： [left_top_landmark_name]     ---例如地标 NO1-L
	# 返回： [array]    ---NO2-L NO2-R NO3-L NO3-R
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称： corners_leveleight
	# 功能： 检测四个角点 ，不是NO1-L NO2-L NO1-R NO1-R，也不是NO1-L NO3-L NO1-R NO3-R，而是NO1-L NO4-L NO1-R NO4-R
	# 初衷： 坐标系标定需要四个角点
	# 状态： 备用
	# 参数： [left_top_landmark_name]     ---例如地标 NO1-L
	# 返回： [array]    ---NO2-L NO2-R NO3-L NO3-R
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def corners_leveleight(self, left_top_landmark_name):
		'''级别8获取角点'''
		labels = [left_top_landmark_name, self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1),
		          self.__fetch_neigbbour(left_top_landmark_name, sourth_step=1, west_step=3),
		          self.__fetch_neigbbour(left_top_landmark_name, west_step=3)]
		return labels

	# ------------------------------------------------
	# 名称： __fetch_neigbbour
	# 功能： 获取邻居
	# 状态： 在用
	# 参数： [landmark_name]     ---例如地标 NO1-L
	#        [sourth_step]       ---例如sourth_step==1，则返回NO1-R
	#        [west_step]        ---例如west_step==1，则返回NO2-L
	# 返回： [landmark_labelname]    ---例如NO2-L
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称： get_next_no
	# 功能： 获取同侧下一个编号
	# 状态： 在用
	# 参数： [landmark_name]     ---例如地标 NO1-L
	#        [forward]       ---例如forward==True,则返回NO0-L，但NO0-L不存在，例如forward==False,则返回NO2-L
	# 返回： [landmark_labelname]    ---例如NO2-L
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
	def get_next_no(self, landmark_name, forward=False):
		result = re.match(self.landmark_match, landmark_name)

		if result is None:
			return landmark_name

		current_no = int(result.group(1))
		next_no = current_no + 1 if not forward else current_no - 1
		next_landmark = "NO{NO}_{D}".format(NO=next_no, D=result.group(2))
		return next_landmark

	# ------------------------------------------------
	# 名称： get_opposite_landmark
	# 功能： 获取对面地标
	# 状态： 在用
	# 参数： [landmark_name]     ---例如地标 NO1-L
	# 返回： [landmark_labelname]    ---例如NO1-R
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称： compute_miss_landmark_position
	# 功能： 如果需要检测四个地标，但是检测到3个，另外一个就需要计算出来
	# 状态： 在用
	# 参数： [landmark_name]     ---例如地标 NO1-L
	# 返回： [（x,y)]    ---例如100,300
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
	def compute_miss_landmark_position(self, landmark_name):
		opposite = self.get_opposite_landmark(landmark_name)
		# print("opposite is {}".format(opposite))
		if opposite not in self.ALL_LANDMARKS_DICT:
			raise NotFoundLandMarkException("opposite landmark is not exist")
		# assert opposite in ALL_LANDMARKS_DICT, "opposite landmark is not exist"
		# 获取同行Y坐标
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
				# 获取同列X坐标
				x = self.ALL_LANDMARKS_DICT[label].col
				break
			except:
				index += 1
				continue

		return x, y

	# ------------------------------------------------
	# 名称： choose_best_cornors
	# 功能： 获取最佳角点列表
	# 状态： 在用
	# 返回： [array]    ---例如[(100,300),(100,500),(500,300),(500,500)]
	# 作者：王杰  编写 2020-4-xx  修改 2020-5-xx
	# ------------------------------------------------
	def choose_best_cornors(self):

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
				# print("label {} need compute".format(label))
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
				if key_j != key and abs(x - xj) < 50 and abs(y - yj) < 50:
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
			if score > 100:
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

	# ------------------------------------------------
	# 名称：  position_loss
	# 功能： 计算丢失率
	# 状态： 在用
	# 参数： [best_four_label_choose]      ---4级地标名称
	#        [best_six_label_choose]       ---6级地标名称
	#        [find]                        ---发现地标与否
	#        [level_four_min_loss]         ---4级地标丢失率
	#        [level_six_min_loss]          ---6级地标丢失率
	# 返回： [best_four_label_choose]      ---例如NO1-L
	#       [best_six_label_choose]        ---例如NO1-L
	#       [loss]       ---例如{4：1,6:1,8:3}
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
	def position_loss(self, best_four_label_choose, best_six_label_choose, find, level_four_min_loss,
	                  level_six_min_loss):
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

	# ------------------------------------------------
	# 名称：  position_landmark
	# 功能： 定位地标
	# 状态： 在用
	# 参数： [image]      ---输入图像
	# 返回： [dest]      ---输出图像，对地标做了标注
	#       [success]        ---地标成功与否，布尔类型
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-xx
	# ------------------------------------------------
	def position_landmark(self, image):
		self.ALL_LANDMARKS_DICT.clear()
		self.ALL_POSITIONS.clear()
		del self.rois
		# start = time.perf_counter()
		rows, cols, channels = image.shape
		if rows != IMG_HEIGHT or cols != IMG_WIDTH:
			dest = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
		else:
			dest = image
		# 地标检测如果出现异常，直接返回
		self.candidate_landmarks(dest, left_start=110, left_end=210, right_start=510, right_end=600)
		for label, (x, y) in self.ALL_POSITIONS.items():
			label_match_result = re.match(self.landmark_match, label)
			no = label_match_result.group('NO')
			exception_labels = [landmarkname for landmarkname, (itemx, itemy) in self.ALL_POSITIONS.items() if
			                    (re.match(self.landmark_match, landmarkname).group('NO') < no and itemy > y) or (
					                    re.match(self.landmark_match, landmarkname).group('NO') > no and itemy < y)]
			if len(exception_labels) > 0:
				return dest, False

		if len(self.ALL_LANDMARKS_DICT.keys()) < 3:
			logger("self.ALL_LANDMARKS_DICT.keys() < 3", level='debug')
			return dest, False

		real_positions = self.__landmark_position_dic()
		real_position_dic = {key: [int(float(value.split(',')[0])), int(float(value.split(',')[1]))] for key, value in
		                     real_positions.items()}

		for landmark_roi in self.rois:
			landmark = landmark_roi.landmark
			if landmark is None or landmark_roi.label not in self.ALL_LANDMARKS_DICT:
				# self.logger("{} miss landmark".format(landmark_roi.label), "warn")
				continue

			col = landmark.col
			row = landmark.row
			real_col, real_row = real_position_dic[landmark_roi.label]
			cv2.rectangle(dest, (col, row), (col + landmark.width, row + landmark.height), color=(255, 0, 255),
			              thickness=2)
			cv2.putText(dest,
			            "({},{})".format(real_col, real_row),
			            (col, row + 90),
			            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
			cv2.putText(dest,
			            "{}".format(landmark_roi.label),
			            (col, row + 60),
			            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
		# 获取最佳的四个地标，如果缺失一个可以通过计算获取
		position_dic, success = self.choose_best_cornors()

		if success:
			# 对检测到的地标，做透视变换
			dest, success = self.__perspective_transform(dest, position_dic)
		# end = time.perf_counter()
		# print("结束{}".format(end - start))
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

	# 绘制网格
	def draw_grid_lines(self, img):
		H_rows, W_cols = img.shape[:2]
		for row in range(0, H_rows):
			if row % 100 == 0:
				cv2.line(img, (0, row), (W_cols, row), color=(255, 255, 0), thickness=1, lineType=cv2.LINE_8)
		for col in range(0, W_cols):
			if col % 50 == 0:
				cv2.line(img, (col, 0), (col, H_rows), color=(255, 255, 0), thickness=1, lineType=cv2.LINE_8)

	# ------------------------------------------------
	# 名称：  __perspective_transform
	# 功能： 透视变换
	# 状态： 在用
	# 注意事项：传递过来的坐标，需要排序
	# 参数： [src]               ---输入图像
	#        [position_dic]      ---坐标字典
	# 返回： [dest]      ---输出图像，重写更新了坐标系
	#       [success]        ---地标成功与否，布尔类型
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def __perspective_transform(self, src, position_dic):
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

		# 对传递过来的坐标值进行排序
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
			dst = cv2.warpPerspective(src, M, (H_rows, W_cols))
		return dst, True

	# ------------------------------------------------
	# 名称：  __compare_hsv_similar
	# 功能： 比较hsv图像相似度
	# 状态： 在用
	# 参数： [img1]               ---输入图像
	#        [img2]               ---输入图像
	# 返回： [数值]               ---相似度
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
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

	# ------------------------------------------------
	# 名称：  find_landmark
	# 功能：  找到地标
	# 注意：相似度方式误差很大，需要指定范围减少计算是关键
	# 参数： [landmark_roi]               ---地标roi
	#        [slide_window_obj]           ---可能坐标位置
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def find_landmark(self, landmark_roi: LandMarkRoi, slide_window_obj: NearLandMark):
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

	# ------------------------------------------------
	# 名称：  candidate_landmarks
	# 功能：  寻找候选地标区域
	# 注意：相似度方式误差很大，需要指定范围减少计算是关键
	# 参数： [dest]               ---输入图像
	#        [left_start]           ---左侧区域开始
	#        [left_end]           ---左侧区域结束
	#        [right_start]           ---可能坐标位置
	#        [right_end]           ---可能坐标位置
	# 作者：王杰  编写 2020-5-xx  修改 2020-5-xx
	# ------------------------------------------------
	def candidate_landmarks(self, dest=None, left_start=110, left_end=210, right_start=510, right_end=600):
		'''left_start=120, left_end=260, right_start=550, right_end=700'''
		global rows, cols, step

		# 不要忽略缩小图片尺寸的重要性，减小尺寸，较少像素数就可以最大限度的减少无用操作；
		# 限制程序速度的最主要因素就是无用操作，无用操作越少，程序执行速度就越高。
		target = dest
		# HSV对光线较RGB有更好的抗干扰能力
		target_hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
		# cv2.imshow("target", target)
		gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
		img_world = np.ones_like(gray)
		ret, img_world = cv2.threshold(img_world, 0, 255, cv2.THRESH_BINARY)

		def warp_filter(c):
			'''内部过滤轮廓'''
			isbig = 200 <= cv2.contourArea(c) < 3600
			# rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
			# return isbig and 3 < rect_w <= 60 and 3 < rect_h <= 60
			return isbig

		def set_mask_area(x: int, y: int, width: int, height: int):
			img_world[y:y + height, x:x + width] = 0

		# 设置边界，边界之外为0
		left_open_mask = np.zeros_like(gray)
		left_open_mask[0:IMG_HEIGHT, left_start:left_end] = 255

		right_open_mask = np.zeros_like(gray)
		right_open_mask[0:IMG_HEIGHT, right_start:right_end] = 255

		bigest_h, bigest_w = 0, 0

		landmarks = []
		for roi_template in self.rois:
			# img_roi_hsvt = cv2.cvtColor(roi_template.roi, cv2.COLOR_BGR2HSV)
			# cv2.imshow("roihist",img_roi_hsvt)
			# img_roi_hsvt = img_roi_hsvt
			# roihist = cv2.calcHist([img_roi_hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
			# cv2.normalize(roihist, roihist, 0, 256, cv2.NORM_MINMAX)
			# 检测二值图像
			foreground = MvSuply.FIND_IT(target, roi_template.roi)
			landmarks.append(foreground)

			# cv2.imshow("landmark", foreground)

			if roi_template.label.find("L") > 0:
				foreground = cv2.bitwise_and(foreground, foreground, mask=left_open_mask)
			if roi_template.label.find("R") > 0:
				foreground = cv2.bitwise_and(foreground, foreground, mask=right_open_mask)

			foreground = cv2.bitwise_and(foreground, foreground, mask=img_world)
			# 图像腐蚀，去除部分噪声，图像中卷积操作无处不在的，只不过没有必要浪费过多精力，用在创造性上更有价值
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			foreground = cv2.dilate(foreground, kernel)
			disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			foreground = cv2.filter2D(foreground, -1, disc)
			ret, foreground = cv2.threshold(foreground, 110, 255, cv2.THRESH_BINARY)

			contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			contours = list(filter(lambda c: warp_filter(c), contours)) if len(contours) > 1 else contours

			if contours is None or len(contours) == 0:
				continue

			# 最有价值的部分在这里，比如检测到了坐标NOL-1,那么地标外延区域就没有必要检测其他地标了，省下的计算资源可是很大
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


# ------------------------------------------------
# 名称：LasterDetector
# 功能：激光灯检测算法.
# 状态：可能不再使用
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class LasterDetector(BaseDetector):

	def __init__(self):
		super().__init__()
		self.laster = None

	# 定位激光灯
	def location_laster(self, img_show, img_copy, middle_start=120, middle_end=450):

		def __filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			logger("laster is {}".format(area), 'info')

			if w < 3 and h < 3 and 1 < area < 5:
				return True
			else:
				return False

		foregroud, contours = self.green_contours(img_copy, middle_start, middle_end)
		# cv2.imshow("green", foregroud)
		# contours = list(filter(__filter_laster_contour, contours))

		if contours is None or len(contours) == 0 or len(contours) > 1:
			return None, foregroud

		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)

		try:
			self.laster = Laster(contours[0], id=0)
			self.laster.modify_box_content()
		except Exception as e:
			mylog_error("laster contour is miss")

		return self.laster, foregroud


# ------------------------------------------------
# 名称：HockDetector
# 功能：钩子检测算法.
# 状态：可能不再使用
# 作者：王杰  编写 2020-3-xx  修改 2020-6-xx
# ------------------------------------------------
class HockDetector(BaseDetector):

	def __init__(self):
		super().__init__()
		self.hock = None
		self._roi = None
		self.__hock_sub_mog = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=36, detectShadows=False)
		self.mask = None

	# ------------------------------------------------
	# 名称：find_edige
	# 功能： 这个方法挺有意思，但是没有成功，做透视变换的时候，图像会有一部分缩减，图像部分区域会成黑色，
	#		 这个方法就是为了检测到黑色边界
	# 状态：未使用
	# 参数：  [dest]            ---输入图像
	# 返回： [底部行号]     ---底部黑色边界
	#        [右侧列号]     ---右侧黑色边界
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_edige(self, dest):
		# 找到左侧边界
		bottom_edge = 0
		for row in range(IMG_HEIGHT // 2, IMG_HEIGHT):
			value = sum([dest[row, i, 0] for i in range(IMG_WIDTH // 2, IMG_WIDTH)])
			if value == 0:
				print("{} 为0".format(row))
				bottom_edge = row
				break

		# 找到右侧边界
		right_edge = 0
		for col in range(IMG_WIDTH // 2, IMG_WIDTH):
			value = sum([dest[i, col, 0] for i in range(IMG_HEIGHT // 2, IMG_HEIGHT)])
			if value == 0:
				right_edge = value
				break

		return bottom_edge, right_edge

	# ------------------------------------------------
	# 名称：find_move_foregrond_method1
	# 功能： 检测运动目标
	# 状态： 使用
	# 参数：  [img]            ---输入图像
	# 返回： [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_move_foregrond_method1(self, img):
		foreground = self.__hock_sub_mog.apply(img)
		return foreground

	# ------------------------------------------------
	# 名称：find_move_foregrond_method2
	# 功能： 检测运动目标，这种方法很简单，缺点非常明显，钩子是运动的，
	#        万一停下了，照样检测不出来
	# 状态： 使用
	# 参数：  [img]            ---输入图像
	# 返回： [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_move_foregrond_method2(self, img):
		frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if not hasattr(self, 'one_frame') or not hasattr(self, 'two_frame') or not hasattr(self,
		                                                                                   'three_frame') or self.one_frame is None or self.two_frame is None or self.three_frame is None:
			self.one_frame = np.zeros_like(frame_gray)
			self.two_frame = np.zeros_like(frame_gray)
			self.three_frame = np.zeros_like(frame_gray)

		# if self.mask is None:
		# 	self.mask = np.zeros_like(frame_gray)

		self.one_frame, self.two_frame, self.three_frame = self.two_frame, self.three_frame, frame_gray
		# if self.one_frame.shape !=self.two_frame.shape:
		# 	return None
		abs1 = cv2.absdiff(self.one_frame, self.two_frame)  # 相减
		_, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

		# if self.two_frame.shape !=self.three_frame.shape:
		# 	return None
		abs2 = cv2.absdiff(self.two_frame, self.three_frame)
		_, thresh2 = cv2.threshold(abs2, 40, 255, cv2.THRESH_BINARY)

		foreground = cv2.bitwise_and(thresh1, thresh2)  # 与运算
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		#
		# foreground[0:IMG_HEIGHT,0:110]=0
		# foreground[0:IMG_HEIGHT, 450:] = 0
		foreground = cv2.filter2D(foreground, -1, kernel)

		#
		# dilate = cv2.dilate(foreground, kernel)  # 膨胀
		# erode = cv2.erode(dilate, kernel)  # 腐蚀
		# dilate = cv2.dilate(dilate, kernel)  # 膨胀

		# contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
		#                                  method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

		# cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for contour in contours:
		# 	if 100 < cv2.contourArea(contour) < 40000:
		# 		x, y, w, h = cv2.boundingRect(contour)  # 找方框
		# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

		return foreground

	# ------------------------------------------------
	# 名称：find_move_foregrond_method3
	# 功能： 检测运动目标，这种方法很简单，缺点非常明显，钩子是运动的，
	#        万一停下了，照样检测不出来
	# 状态： 使用
	# 参数：  [img]            ---输入图像
	# 返回： [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_move_foregrond_method3(self, img):
		'''
		find foreground by diff

		:return:
		'''
		frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if not hasattr(self, 'one_frame') or not hasattr(self,
		                                                 'two_frame') or self.one_frame is None or self.two_frame is None:
			self.one_frame = np.zeros_like(frame_gray)
			self.two_frame = np.zeros_like(frame_gray)

		# if self.mask is None:
		# 	self.mask = np.zeros_like(frame_gray)

		self.one_frame, self.two_frame = self.two_frame, frame_gray
		# if self.one_frame.shape !=self.two_frame.shape:
		# 	return None
		abs1 = cv2.absdiff(self.one_frame, self.two_frame)  # 相减
		_, thresh1 = cv2.threshold(abs1, 40, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		#
		# foreground[0:IMG_HEIGHT,0:110]=0
		# foreground[0:IMG_HEIGHT, 450:] = 0
		foreground = cv2.filter2D(thresh1, -1, kernel)

		#
		# dilate = cv2.dilate(foreground, kernel)  # 膨胀
		# erode = cv2.erode(dilate, kernel)  # 腐蚀
		# dilate = cv2.dilate(dilate, kernel)  # 膨胀

		# contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
		#                                  method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

		# cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# for contour in contours:
		# 	if 100 < cv2.contourArea(contour) < 40000:
		# 		x, y, w, h = cv2.boundingRect(contour)  # 找方框
		# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

		return foreground

	# ------------------------------------------------
	# 名称：find_green_contours
	# 功能： 检测绿色
	# 状态： 使用
	# 参数：  [img]            ---输入图像
	#         [middle_start]   ---开始边界
	#         [middle_end]     ---结束边界
	# 返回： [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def find_green_contours(self, img, middle_start=120, middle_end=450):
		foreground, contours = self.green_contours(img, middle_start=middle_start, middle_end=middle_end)
		return foreground

	@property
	def hock_roi(self):
		if self._roi is None:
			hock_rois = [
				HockRoi(img=cv2.imread(os.path.join(HOCK_ROI, roi_img)))
				for
				roi_img in
				os.listdir(HOCK_ROI)]
			if len(os.listdir(HOCK_ROI)) == 0:
				self._roi = None
			self._roi = hock_rois[0]
		return self._roi

	# ------------------------------------------------
	# 名称：hock_foreground
	# 功能： 读取钩子二值图像
	# 状态： 使用
	# 参数：  [img_copy]            ---输入图像
	#         [middle_start]   ---开始边界
	#         [middle_end]     ---结束边界
	# 返回： [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def hock_foreground(self, img_copy, middle_start=120, middle_end=470):
		# target_hsvt = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
		# img_roi_hsvt = cv2.cvtColor(self.hock_roi.img, cv2.COLOR_BGR2HSV)
		# # cv2.imshow("roihist",img_roi_hsvt)
		# img_roi_hsvt = img_roi_hsvt
		# roihist = cv2.calcHist([img_roi_hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
		# cv2.normalize(roihist, roihist, 0, 256, cv2.NORM_MINMAX)
		#
		# foreground = self.find_it(images=[target_hsvt], hist=roihist)

		# foreground = self.find_move_foregrond_method3(img_copy)

		# move_foreground = self.find_move_foregrond_method1(img_copy)

		# mask = np.zeros_like(move_foreground)
		# mask[0:IMG_HEIGHT, middle_start:middle_end] = 255

		# 检测定位钩方法
		yellow_foreground, _d = self.yellow_contours(img_copy)

		# foreground = cv2.bitwise_and(move_foreground, yellow_foreground, mask)

		# foreground=cv2.bitwise_or(move_foreground,yellow_foreground,mask)
		foreground = yellow_foreground

		# bottom_edge,right_edge=self.find_edige(img_copy)
		foreground[0:IMG_HEIGHT, 0:middle_start] = 0
		foreground[0:IMG_HEIGHT, middle_end:] = 0

		# gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
		# middle_mask = np.zeros_like(gray)
		# middle_mask[0:IMG_HEIGHT, middle_start:middle_end] = 255
		# middle_mask[0:bottom_edge, middle_start:middle_end] = 255
		# result_foreground=cv2.bitwise_and(foreground1, foreground1, middle_mask)

		contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return contours, foreground

	# ------------------------------------------------
	# 名称：location_hock_withlandmark
	# 功能： 读取钩子二值图像
	# 状态： 使用
	# 参数：  [img_show]            ---输入图像
	#         [img_copy]   ---开始边界
	#         [find_landmark]     ---结束边界
	#         [middle_start]   ---开始边界
	#         [middle_end]     ---结束边界
	# 返回： [钩子对象]     ---钩子对象
	#       [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_hock_withlandmark(self, img_show, img_copy, find_landmark: bool, middle_start=130, middle_end=470):

		def if_rectangle(c):
			# 初始化形状名和近似的轮廓
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)
			return len(approx) == 4

		# 定位钩轮廓过滤条件：边界、宽度、高度、轮廓角点数目、相似度作为限制
		def __filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			if x < middle_start or x > 500:
				return False
			roi_img = cv2.resize(self.hock_roi.img, (w, h))
			radio = self.color_similar_ratio(roi_img, img_show[y:y + h, x:x + w])
			if w < 7 or h < 7:
				# if radio>0:print("radio:{},w:{},h:{}".format(radio, w, h))
				return False
			if w > 45 or h > 45:
				# if radio>0:print("radio:{},w:{},h:{}".format(radio, w, h))
				return False
			if radio < 0:
				return False

			if if_rectangle(c) or radio > 0.21:
				if 7 <= w < 45 and 7 <= h < 45 and radio > 0.04:
					return True
				else:
					# print("radio:{},w:{},h:{}".format(radio, w, h))
					return False

			# print("wide is {},height is {}".format(w, h))
			return False

		# 定位钩前景目标
		contours, foregroud = self.hock_foreground(img_copy, middle_start, middle_end)
		# green_ret, foreground = cv2.threshold(foregroud, 40, 255, cv2.THRESH_BINARY)
		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foregroud = cv2.filter2D(foregroud, -1, disc)
		contours = list(filter(__filter_laster_contour, contours))

		if contours is None or len(contours) == 0 or len(contours) > 1:
			return None, foregroud
		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)
		try:
			self.hock = Hock(contours[0], id=0)
			# self.hock.modify_box_content()
			cv2.putText(img_show, self.hock.box_content,
			            (self.hock.boxcenterpoint[0], self.hock.boxcenterpoint[1]),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		# print(self.hock.img_x,self.hock.img_y)
		except Exception as e:
			mylog_error("hock contour is miss")

		return self.hock, foregroud

	# ------------------------------------------------
	# 名称：location_hock_withoutlandmark
	# 功能： 读取钩子二值图像
	# 注意：1、去除前景目标边框噪音及移动造成的空洞噪音
	# 		2、钩子多方法融合，根据检测点区域检测出来
	# 状态： 使用
	# 参数：  [img_show]            ---输入图像
	#         [middle_start]   ---开始边界
	#         [middle_end]     ---结束边界
	# 返回： [钩子对象]     ---钩子对象
	#       [foreground]     ---二值图像
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-xx
	# ------------------------------------------------
	def location_hock_withoutlandmark(self, img_show, middle_start=245, middle_end=490):

		def if_rectangle(c):
			# 初始化形状名和近似的轮廓
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.04 * peri, True)
			return len(approx) == 4

		def __filter_laster_contour(c):
			x, y, w, h = cv2.boundingRect(c)
			area = cv2.contourArea(c)
			# center_x, center_y = (x + round(w * 0.5), y + round(h * 0.5))
			if x < middle_start or x > 500:
				return False
			roi_img = cv2.resize(self.hock_roi.img, (w, h))
			radio = self.color_similar_ratio(roi_img, img_show[y:y + h, x:x + w])
			if w < 7 or h < 7:
				# if radio>0:print("radio:{},w:{},h:{}".format(radio, w, h))
				return False
			if w > 45 or h > 45:
				# if radio>0:print("radio:{},w:{},h:{}".format(radio, w, h))
				return False
			if radio < 0:
				return False

			if if_rectangle(c) or radio > 0.21:
				if 7 <= w < 45 and 7 <= h < 45 and radio > 0.04:
					return True
				else:
					# print("radio:{},w:{},h:{}".format(radio, w, h))
					return False

			# print("wide is {},height is {}".format(w, h))
			return False

		contours, foregroud = self.hock_foreground(img_show, middle_start, middle_end)
		# green_ret, foreground = cv2.threshold(foregroud, 40, 255, cv2.THRESH_BINARY)
		# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		# foregroud = cv2.filter2D(foregroud, -1, disc)
		contours = list(filter(__filter_laster_contour, contours))

		cv2.drawContours(img_show, contours, -1, (255, 0, 0), 3)
		# 定位钩是唯一的，出来多个轮廓，就是错误的情况
		if contours is None or len(contours) == 0 or len(contours) > 1:
			return None, foregroud

		try:
			hock = Hock(contours[0], id=0)
			hock.modify_box_content()
			cv2.putText(img_show, hock.box_content,
			            (hock.boxcenterpoint[0], hock.boxcenterpoint[1]),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)
		# print(self.hock.img_x,self.hock.img_y)
		except Exception as e:
			mylog_error("hock contour is miss")

		return hock, foregroud


if __name__ == '__main__':
	pass

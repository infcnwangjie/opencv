# -*- coding: utf-8 -*-
import math
import os
import time
from collections import defaultdict
from time import sleep

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from app.config import IMG_WIDTH, IMG_HEIGHT, SAVE_VIDEO_DIR, HOCK_DISTANCE, DEBUG
from app.core.autowork.detector import LandMarkDetecotr, BagDetector, LasterDetector, HockDetector

from app.log.logtool import logger


# ------------------------------------------------
# 名称：DetectorHandle
# 功能：检测句柄，作为SERVICE层使用，供线程调用使用
# 状态：在用，后期重构之后会改动
# 作者：王杰  2020-4-15
# ------------------------------------------------
class DetectorHandle(object):
	instance = None
	current_target_x, current_target_y = 0, 0  # 当前目标袋子坐标 x,y
	current_hock_x, current_hock_y = 0, 0  # 当前钩子坐标 x,y
	error_x, error_y = 0, 0  # 当前x坐标纠偏，当前y坐标纠偏
	input_move_instructs = []  # 所有的移动指令
	hock_points = []

	def __new__(cls, *args, **kwargs):
		if cls.instance is None:
			cls.instance = super().__new__(cls)
		return cls.instance

	def __init__(self, plchandle):
		self.bags = []  # 袋子列表
		self.current_bag = None  # 当前处理的袋子
		self.hock = None  # 钩子
		self.last_hock = None  # 最近一次检测到的hock
		self.laster = None  # 激光灯
		self.last_laster = None  # 最近一次检测到的激光灯
		self.finish_job = False  # 完成了所有搬运工作
		self.landmark_detect = LandMarkDetecotr()
		self.bag_detect = BagDetector()
		self.laster_detect = LasterDetector()
		self.hock_detect = HockDetector()
		self.history_bags = []
		self.history_laster_travel = []
		self.plchandle = plchandle
		self.status_show = None  # 状态栏更新消息

	# ------------------------------------------------
	# 名称：plc_connect
	# 功能：检测PLC是否连接
	# 状态：在用
	# 参数： [None]   ---
	# 返回： [布尔]   ---连接与否
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def plc_connect(self):
		plcconnect = self.plchandle.is_open()
		return plcconnect if not DEBUG else True

	# ------------------------------------------------
	# 名称：compute_img
	# 功能：处理视频帧处理
	# 状态：在用
	# 参数： [show]   ---输入图像
	#        [index]   ---当前帧数
	# 返回： [image]   ---检测地标成功返回透视变换后的图片，否则返回未透视化的
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def compute_img(self, show, index):
		plc_connect = self.plc_connect()

		if plc_connect == False:
			logger("PLC连接失败", level='info')
			cv2.putText(show, "plc connect fail", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
			return show

		perspective_img, find_landmark = self.landmark_detect.position_landmark(show)

		# 地标定位失败，就下一帧
		if find_landmark == False:
			bag_info = self.get_currentbag_position_withoutlandmark(show)
			hock_info = self.get_hock_position_withoutlandmark(show)
			if self.current_bag is not None and self.current_bag.status_map[
				'drop_hock'] == True and self.current_bag.step == 'drop_hock':
				self.suck_bag(perspective_img=None, original_img=show, frameindex=index)
			self.vision_no_landmark(msg="没有发现地标，向东移动1米")
			return show
		else:
			# 计算最近的袋子,就能与钩子最近的袋子找到
			self.update_hockposition(perspective_img, original_img=None, find_landmark=find_landmark)
			# 如果当前袋子为空,或者当前袋子执行完毕,需要切换目标袋子
			self.choose_or_update_currentbag(perspective_img, original_img=None, find_landmark=find_landmark)

			if self.current_bag is not None and self.hock is not None:
				# 这里并没有书写错误，为了防止坐标误差，直接拿原图像判断是否吸住
				self.move_or_suck(perspective_img=perspective_img, original_img=show, frameindex=index)
			else:
				# 行车与目标距离较远，没有识别到
				self.vision_no_bag(msg="没有发现袋子，向东移动1米")

			return perspective_img

	# ------------------------------------------------
	# 名称：move_or_suck
	# 功能：处理视频帧处理
	# 状态：在用
	# 参数： [perspective_img]   ---透视变化图像
	#        [original_img]   ---未透视变化的图像
	#        [frameindex]   ---当前帧数
	# 返回： [None]   ---定位钩与目标袋子位移太大时，运动；否则，降钩并且处理落钩问题
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def move_or_suck(self, perspective_img, original_img, frameindex):

		# print("hock position({},{}),bag position({},{})".format(self.hock.x, self.hock.y, self.current_bag.x,
		#                                                         self.current_bag.y))
		if abs(
				self.current_bag.x - self.hock.x) > 50 or abs(
			self.current_bag.y - self.hock.y - HOCK_DISTANCE) > 50:
			self.move_to_nearestbag(perspective_img)
		else:

			if self.current_bag.step == 'move_close':
				self.current_bag.step = 'drop_hock'
				self.current_bag.status_map['drop_hock'] = True
			if self.current_bag.step == 'drop_hock' and self.current_bag.status_map['drop_hock'] == True:
				self.suck_bag(perspective_img=None, original_img=original_img, frameindex=frameindex)

	# ------------------------------------------------
	# 名称：vision_no_landmark
	# 功能：目前视野中没有指定目标
	# 状态：在用
	# 参数： [msg]    ---反馈信息
	# 返回： [None]   ---看不到地标或者看不到目标，是因为还没有进入视野
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def vision_no_landmark(self, msg=None):
		if DEBUG:
			return
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()
			# 确保行车没有被紧急停止或者没有在运动状态中
			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')
		# laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		#                                                                middle_end=450)
		# self.ugent_stop_car(laster.x, laster.y + 50, 0 )
		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	# ------------------------------------------------
	# 名称：vision_no_bag
	# 功能：目前视野中没有指定袋子
	# 状态：在用
	# 参数： [msg]    ---反馈信息
	# 返回： [None]   ---看不到地标或者看不到目标，是因为还没有进入视野
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def vision_no_bag(self, msg=None):
		if DEBUG:
			return
		try:
			move_status = self.plchandle.read_status()
			is_ugent_stop = self.plchandle.is_ugent_stop()
			# 确保行车没有被紧急停止或者没有在运动状态中
			if move_status == 0 and is_ugent_stop == 0:
				self.plchandle.move(east=100)
				if msg is not None:
					logger(msg, 'info')
		# laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		#                                                                middle_end=450)
		# self.ugent_stop_car(laster.x, laster.y + 50, 0 )
		except Exception as e:
			logger("plc没有开启或者连接失败", "error")

	# ------------------------------------------------
	# 名称：find_laster
	# 功能：定位激光灯
	# 状态：在用
	# 参数： [dest]             ---输入图像
	#        [find_landmark]    ---是否发现地标
	# 返回： [dest]   ---dest中有激光灯的轮廓
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def find_laster(self, dest, find_landmark=False):

		dest_copy = dest.copy()
		laster, laster_foreground = self.laster_detect.location_laster(dest, dest_copy, middle_start=100,
		                                                               middle_end=450)
		if laster is not None and find_landmark == True:
			self.last_laster = laster
		self.laster = laster
		return dest

	# ------------------------------------------------
	# 名称：hock_moveto_center
	# 功能：启动行车时候，先将行车移到中间区域X=center，防止因边界条件检测不到定位钩
	# 状态：在用
	# 参数： [None]             ---
	# 返回： [None]   ---移动钩子到行车中间区域X=center
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def hock_moveto_center(self):
		try:
			self.plchandle.move(nourth=2)
		except:
			logger("plc is not connect", level="error")

	# ------------------------------------------------
	# 名称：update_hockposition
	# 功能：实时更新定位钩坐标
	# 状态：在用
	# 参数： [perspective_img]          ---透视图像
	#        [original_img]             ---未透视的图像
	#        [find_landmark]            ---是否定位地标成功
	# 返回： [hock_x]   ---定位钩X坐标
	#       [hock_y]   ---定位钩Y坐标
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def update_hockposition(self, perspective_img=None, original_img=None, find_landmark=False):
		if find_landmark == False:
			return None
		perspective_img_copy = perspective_img.copy()

		hock, hock_foreground = self.hock_detect.location_hock_withlandmark(perspective_img, perspective_img_copy,
		                                                                    find_landmark,
		                                                                    middle_start=135,
		                                                                    middle_end=465)
		if hock is None:
			return None
		self.last_hock = hock
		if self.hock is None or not hasattr(self, 'hock'):
			self.hock = hock
		else:
			self.hock.x, self.hock.y = hock.x, hock.y

	# ------------------------------------------------
	# 名称：get_currentbag_position_withoutlandmark
	# 功能：更新袋子坐标，仅限于地标定位失败之时
	# 状态：在用
	# 参数： [original_img]          ---未透视的图像
	# 返回： [bag_x]   ---选择袋子X坐标
	#       [bag_y]   ---选择袋子Y坐标
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def get_currentbag_position_withoutlandmark(self, original_img=None):

		bags, _foreground = self.bag_detect.location_bags_withoutlandmark(original_img)
		if bags is None or self.current_bag is None:
			return None
		choose_bag = list(filter(lambda bag: bag.id == self.current_bag.id, bags))
		if choose_bag is None or len(choose_bag) == 0:
			return None
		else:
			return choose_bag[0].x, choose_bag[0].y

	# ------------------------------------------------
	# 名称：get_hock_position_withoutlandmark
	# 功能：更新钩子坐标，仅限于地标定位失败之时
	# 状态：在用
	# 参数： [original_img]          ---未透视的图像
	# 返回： [hock_x]   ---定位钩X坐标
	#       [hock_y]   ---定位钩Y坐标
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def get_hock_position_withoutlandmark(self, original_img=None):
		hock, _foreground = self.hock_detect.location_hock_withoutlandmark(original_img)
		if hock is None:
			return None
		else:
			x, y, w, h = hock.x, hock.y, hock.w, hock.h
			cv2.rectangle(original_img, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 255, 0), 1)
			return hock.x, hock.y

	# ------------------------------------------------
	# 名称：choose_or_update_currentbag
	# 功能：选择并更新当前袋子的坐标
	# 状态：在用
	# 参数： [perspective_img]          ---未透视的图像
	#        [original_img]          ---未透视的图像
	#       [original_img]          ---未透视的图像
	# 返回： [None]   ---设置当前袋子
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def choose_or_update_currentbag(self, perspective_img=None, original_img=None, find_landmark=False):
		if find_landmark == False:
			return None
		if not DEBUG:
			ugent_stop_status = self.plchandle.is_ugent_stop()
		else:
			ugent_stop_status = 0

		# 如果处在紧急停止状态，就不做处理了
		if ugent_stop_status == 1:
			self.landmark_detect.draw_grid_lines(perspective_img if perspective_img is not None else original_img)
			return None

		perspective_copy_img = perspective_img.copy()

		# 定位到地标与定位失败最大的区别就是，开始边界与结束边界不一样，坐标系也不一样
		bags, bag_forground = self.bag_detect.location_bags_withlandmark(perspective_img, perspective_copy_img,
		                                                                 find_landmark,
		                                                                 middle_start=120,
		                                                                 middle_end=500)
		self.bags = bags

		# 每个袋子有唯一ID
		if self.current_bag is not None:
			for bag in self.bags:
				if bag.id == self.current_bag.id and self.current_bag.status_map['finish_move'] == False:
					self.current_bag = bag

		if self.current_bag and self.current_bag.status_map['finish_move'] == False:
			return None

		if bags is None or len(bags) == 0:
			if self.hock is not None:
				if self.hock.y < 100:
					self.finish_job = True  # 结束自动运输任务
					self.plchandle.ugent_stop = True  # 越界紧急停止运动
					self.plchandle.power = False  # 行车梯形图断电
			else:
				# 袋子检测失败
				self.vision_no_landmark(msg="没有发现袋子，向东移动一米")
				self.landmark_detect.draw_grid_lines(perspective_img)
			return None
		else:
			# 如果所有袋子都处理完了，PLC梯形图电源要关闭，通过寄存器地址传输信息
			need_process_bags = [bag for bag in self.bags if bag.status_map['finish_move'] == False]
			if need_process_bags is None or len(need_process_bags) == 0:
				# 没有要移动的袋子,让plc关闭梯形图程序
				self.finish_job = True
				self.plchandle.power = False
				return None

		if self.hock is None:
			# 当前帧，钩子定位失败
			self.landmark_detect.draw_grid_lines(perspective_img if perspective_img is not None else original_img)
			return None

		# 如果当前没有需要运行的袋子，或者当前目标袋子已经处理完成，则换一个袋子
		if self.current_bag is None or self.current_bag.status_map['finish_move'] == True:
			choose_index = self.choose_nearest_bag(self.bags, self.hock)
			choosed_bag = bags[choose_index]
			self.current_bag = choosed_bag
			self.current_bag.status_map['choose'] = True
			self.current_bag.step = 'choose'
			logger("find nearest bag->({},{})".format(choosed_bag.x, choosed_bag.y), level='info')

	# ------------------------------------------------
	# 名称：down_hock
	# 功能：放下定位钩，正常情况，定位钩与真实钩子同时下落
	# 状态：在用
	# 参数： [much]          ---下降步数为50公分
	# 返回： [None]   ---下落钩子
	# 作者：王杰  2020-6-xx
	# ------------------------------------------------
	def down_hock(self, much=50):
		self.plchandle.move(down=much)
		if self.current_bag is not None:
			self.current_bag.down_hock_much += much

	# ------------------------------------------------
	# 名称：check_suck
	# 功能：检测定位钩是否吸住袋子
	# TODO  目前还在做着按照下面5个方向逐渐试试
	# 方向：1、定位钩会一直下降，下降的过程中，定位钩会做一定幅度的摆动
	# 		2、定位钩下降过程中，如果摆动小了或者没有摆动，可能就吸住了
	# 		3、定位钩与袋子的坐标一致
	# 		4、混动定位钩时，袋子是否发生位移
	# 		5、难点：磁铁吸住袋子的时候，地标检测失败，程序丢失信息，因此改用original_img
	# 状态：在用
	# 参数： [perspective_img]       ---透视变换后的图像
	#        [original_img]          ---未透视变换后的图像
	#        [frame_index]           ---当前帧数
	# 返回： [布尔]   ---是否吸住
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def check_suck(self, perspective_img=None, original_img=None, frame_index=0):

		if self.current_bag is None:
			return False

		# 检测袋子坐标信息
		bag_info = self.get_currentbag_position_withoutlandmark(original_img)
		# 检测钩子坐标信息
		hock_info = self.get_hock_position_withoutlandmark(original_img)
		if bag_info is None or hock_info is None:
			return False

		bag_col, bag_row = bag_info
		hock_col, hock_row = hock_info

		# 只记录坐标误差小于20的钩子坐标信息
		if abs(hock_col - bag_col) < 20 and abs(
				hock_row - bag_row) < 20:
			self.current_bag.suckhock_positions[str(frame_index)] = hock_info
		else:
			self.current_bag.suckhock_positions[str(frame_index)] = None

		if len(self.current_bag.suckhock_positions.keys()) < 10:
			return False

		suck_success = False
		suck_time_tj = defaultdict(int)
		keep_frames = 0
		# 当前帧-10至当前帧内的位置信息
		for index, position in {int(index_str): location for index_str, location in
		                        self.current_bag.suckhock_positions.items() if
		                        frame_index - 10 < int(index_str) < frame_index}.items():
			if position is None:
				continue
			# 如果前一帧为空，但现在帧不为空，那么持续帧置keep_frames=1
			if str(index - 1) not in self.current_bag.suckhock_positions or \
					self.current_bag.suckhock_positions[
						str(index - 1)] is None:
				keep_frames = 1
				continue

			if str(index - 1) in self.current_bag.suckhock_positions and self.current_bag.suckhock_positions[
				str(index - 1)] is not None:

				if position is None:
					# 如果当前帧为空，那么持续帧置keep_frames=1
					suck_time_tj[str(keep_frames)] += 1
					keep_frames = 0
					continue
				else:
					# 如果当前帧不为空，那么持续帧置keep_frames累加，如果累加数大于5就返回TRUE
					col, row = position
					col_last, row_last = self.current_bag.suckhock_positions[
						str(index - 1)]

					if abs(col - col_last) == 0 and abs(row - row_last) == 0:
						keep_frames += 1

						if suck_time_tj['5'] > 1:
							suck_success = True
							break

					else:
						suck_time_tj[str(keep_frames)] += 1
						keep_frames = 0
		else:
			suck_success = False

		print(suck_time_tj)
		if self.status_show is not None:
			self.status_show.showMessage("已经吸住")

		return suck_success

	# ------------------------------------------------
	# 名称：check_hold
	# 功能：检测真实抓手是否抓住袋子
	# TODO  目前靠想象去做
	# 状态：准备用
	# 参数： [perspective_img]       ---透视变换后的图像
	#        [original_img]          ---未透视变换后的图像
	#        [index]                 ---当前帧数
	# 返回： [布尔]   ---是否抓住
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def check_hold(self, perspective_img, original_img, index=0):
		'''
		检测机械手抓住袋子：袋子的面积会因为被机械手遮挡而变小
		:param original_img: 仅仅只是row:900
		:return:
		'''
		# cv2.putText(dest, "check_hold", (360, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
		#             (255, 255, 255), 2)

		# self.suck_times = 0
		if self.status_show is not None:
			self.status_show.showMessage("检测是否钩住袋子")
		return True

	# ------------------------------------------------
	# 名称：pull_bag
	# 功能：检测真实抓手是否抓住袋子
	# TODO  目前靠想象去做
	# 状态：准备用
	# 参数： [None]       ---透视变换后的图像
	# 返回： [布尔]   ---是否拉起
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def pull_bag(self):
		'''
		拉起袋子，放到目的区域，目前没做
		:return:
		'''
		pass

	# ------------------------------------------------
	# 名称：suck_bag
	# 功能：吸住钩子操作
	# 难点：行车下降多少，胡工控制不住，不能我让它下降多少它就下降多少，我还在要求他做到可控！！！！！
	# TODO  目前在做
	# 状态：准备用
	# 参数： [perspective_img]       ---透视变换后的图像
	#        [original_img]          ---未透视变换后的图像
	#        [frameindex]                 ---当前帧数
	# 返回： [None]   ---吸住钩子控制逻辑
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def suck_bag(self, perspective_img=None, original_img=None, frameindex=0):

		if self.current_bag is not None and self.current_bag.status_map['hock_suck'] == True:
			return True

		if perspective_img is not None:
			cv2.putText(perspective_img, "droping hock", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)
		else:
			cv2.putText(original_img, "droping hock", (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

		if self.status_show is not None:
			self.status_show.showMessage("定位钩下降")

		if_suck, max_drop_z, has_droped_z = False, 400, 0  # 捡起
		while if_suck == False and has_droped_z < max_drop_z:
			self.down_hock(50)
			has_droped_z += 50
			if perspective_img is not None:
				self.move_to_nearestbag(perspective_img)
			if_suck = self.check_suck(perspective_img=None, original_img=original_img, frame_index=frameindex)
		if if_suck == True:
			# print(self.current_bag.suck_frame_status)
			self.current_bag.step = 'hock_suck'
			self.current_bag.status_map['hock_suck'] = True
			print(self.current_bag.suckhock_positions)
			if perspective_img is not None:
				cv2.putText(perspective_img, "suck bag success", (500, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			if original_img is not None:
				cv2.putText(original_img, "suck bag success", (500, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)

			if self.status_show is not None:
				self.status_show.showMessage("定位钩已吸住" if if_suck else "未吸住")
			hold_bag = self.check_hold(perspective_img, original_img, frameindex)
			if hold_bag:
				self.pull_bag()

	# ------------------------------------------------
	# 名称：suck_bag
	# 功能：吸住钩子操作
	# TODO  目前在做
	# 状态：准备用
	# 参数： [perspective_img]       ---透视变换后的图像
	#        [original_img]          ---未透视变换后的图像
	#        [frameindex]                 ---当前帧数
	# 返回： [None]   ---吸住钩子控制逻辑
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def move_to_nearestbag(self, perspective_img):
		'''
		钩子向袋子靠近
		:param dest:
		:return:
		'''
		if perspective_img is None:
			return None
		try:
			if not DEBUG:
				move_status = self.plchandle.read_status()
				is_ugent_stop = self.plchandle.is_ugent_stop()
			else:
				move_status = 0
				is_ugent_stop = 0

			# move==1说明行车在移动中，0静止
			if move_status == 1 or is_ugent_stop == 1:
				if len(self.input_move_instructs) > 0:
					cv2.putText(perspective_img, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX,
					            1.2,
					            (255, 255, 255), 2)
				# 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
				self.landmark_detect.draw_grid_lines(perspective_img)
				return perspective_img

			# 视频中行车激光位置，钩子的位置需要定位
			current_car_x, current_car_y, current_car_z = self.hock.x, self.hock.y + HOCK_DISTANCE, 0
			# self.hock_points.append(current_car_x, current_car_y)

			# 写入目标坐标
			target_x, target_y, target_z = self.current_bag.x, self.current_bag.y, 0
			target_info = "bag_X:{},bag_Y:{}".format(target_x, target_y)
			logger(target_info, level='info')
			cv2.putText(perspective_img, target_info, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			east, west, south, north, up, down = 0, 0, 0, 0, 0, 0

			move_info = "I:"
			if target_x - current_car_x > 0:
				south = abs(target_x - current_car_x)
				move_info += "to S {} cm".format(south)
			else:
				north = abs(target_x - current_car_x)
				move_info += "to N {} cm".format(north)

			if target_y - current_car_y > 0:
				west = abs(target_y - current_car_y)
				move_info += ", to W {} cm".format(west)
			else:
				east = abs(target_y - current_car_y)
				move_info += ",to E {} cm".format(east)

			if target_z - current_car_z > 0:
				up = abs(target_z - current_car_z)
				move_info += ",UP {} cm".format(up)
			else:
				down = abs(target_z - current_car_z)
				move_info += ",DOWN {} cm".format(down)

			if not DEBUG:
				self.plchandle.move(east=east, west=west, south=south, nourth=north, up=up, down=down)
			logger(move_info, level='info')
			self.input_move_instructs.append(move_info)

			self.current_bag.status_map['move_close'] = True
			self.current_bag.step = 'move_close'

			# if len(self.input_move_instructs) > 0:
			# 	cv2.putText(perspective_img, self.input_move_instructs[-1], (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			# 	            (255, 255, 255), 2)
			current_hock_info = "HOCK->X:{},Y:{}".format(current_car_x, current_car_y)
			logger(current_hock_info, 'info')
			cv2.putText(perspective_img, current_hock_info, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			error_info = "ERROR:{},{},{}".format(abs(target_x - current_car_x), abs(target_y - current_car_y - 30),
			                                     abs(target_z - current_car_z))
			logger(error_info, level='info')
			cv2.putText(perspective_img, error_info, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
			            (255, 255, 255), 2)

			if not DEBUG:
				# 超出边界就要紧急停止
				if current_car_x < 100 or current_car_x > 550 or current_car_y < 20:
					self.ugent_stop_car(current_car_x, current_car_y, current_car_z, perspective_img)
		except Exception as e:
			print(e)
			self.landmark_detect.draw_grid_lines(perspective_img)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
			return perspective_img
		self.landmark_detect.draw_grid_lines(perspective_img)  # 放到最后是为了防止网格线给袋子以及激光灯的识别带来干扰
		return perspective_img

	# ------------------------------------------------
	# 名称：ugent_stop_car
	# 功能：紧急停止行车
	# 初衷：行车移动过程越界可能会造成危险
	# 状态：在用
	# 参数： [current_car_x]       ---当前行车x坐标
	#        [current_car_y]       ---当前行车y坐标
	#        [current_car_z]       ---当前行车z坐标
	#        [dest]                ---目标图像
	# 返回： [None]   ---紧急停止逻辑
	# 作者：王杰  编写 2020-6-xx  修改 2020-6-12
	# ------------------------------------------------
	def ugent_stop_car(self, current_car_x, current_car_y, current_car_z, dest=None):
		# 智能识别紧急停止行车
		if current_car_y == 0 or current_car_y > 800 or current_car_x == 0 or current_car_x > 500 or current_car_x < 0 or current_car_y < 0:

			if dest is not None:
				cv2.putText(dest, " ugent_stop {},{},{}".format(current_car_x, current_car_y, current_car_z),
				            (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
				            (255, 255, 255), 2)
			self.work = False
			if self.save_video:
				self.update_savevideo.emit(self.save_video_name)
			if not DEBUG:
				self.plchandle.ugent_stop()

	# ------------------------------------------------
	# 名称：choose_nearest_bag
	# 功能：选择距离钩子最近的袋子
	# 初衷：选择距离钩子最近的袋子
	# 状态：在用
	# 参数： [current_car_x]       ---当前行车x坐标
	#        [current_car_y]       ---当前行车y坐标
	#        [current_car_z]       ---当前行车z坐标
	#        [dest]                ---目标图像
	# 返回： [None]   ---紧急停止逻辑
	# 作者：王杰  编写 2020-5-xx  修改 2020-6-12
	# ------------------------------------------------
	def choose_nearest_bag(self, bags, hock):

		def __compute_distance(bag, hock):
			# start= time.perf_counter()
			X_2 = math.pow(bag.x - hock.x, 2)
			Y_2 = math.pow(bag.y - hock.y, 2)
			distance = math.sqrt(X_2 + Y_2)
			# end = time.perf_counter()
			# print("distance is {},compute cost:{}".format(distance,end-start))
			return distance

		# 按照定位钩与每个袋子坐标的误差判断
		distances = [__compute_distance(bag, hock) for bag in bags if bag.status_map['finish_move'] == False]

		min_distance, choose_index = 10000, 0
		for index, d in enumerate(distances):
			if d < min_distance:
				min_distance = d
				choose_index = index
		return choose_index


# ------------------------------------------------
	# 名称：ProcessThread
	# 功能：线程操作在界面编程中是非常实用的
	# 状态：在用
	# 作者：王杰  编写 2020-3-xx  修改 2020-6-12
	# ------------------------------------------------
class ProcessThread(QThread):
	update_savevideo = pyqtSignal(str)

	def __init__(self, video_player, IMGHANDLE=None, PLCHANDLE=None, parent=None):
		super().__init__(parent=parent)
		self._playing = True
		self._finish = False
		self._working = False
		self.video_player = video_player
		self.IMAGE_HANDLE = IMGHANDLE  # 从skd中获取图像
		# self.plchandle = PLCHANDLE
		self.save_video = False
		self.detectorhandle = DetectorHandle(PLCHANDLE)

	def __del__(self):
		self._working = False
		if hasattr(self.IMAGE_HANDLE, 'release') and self.IMAGE_HANDLE:
			self.IMAGE_HANDLE.release()

	@property
	def play(self):
		return self._playing
	# 启动播放
	@play.setter
	def play(self, value=True):
		self._playing = value

	@property
	def work(self):
		return self._working

	# ------------------------------------------------
	# 名称：work
	# 功能：智能抓手自动工作开关
	# 状态：在用
	# 参数： [value]       ---布尔决定是否启动开关
	# 返回： [None]   ---
	# 作者：王杰  编写 2020-4-xx  修改 2020-4-xx
	# ------------------------------------------------
	@work.setter
	def work(self, value=True):
		self._working = value

	def run(self):
		# 启动行车，最好先将定位钩移到中间
		self.detectorhandle.hock_moveto_center()
		save_video_name = time.strftime("%Y%m%d%X", time.localtime()).replace(":", "")
		self.save_video_name = "saved_" + save_video_name + '.avi'
		out = None
		if self.save_video:
			fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 保存视频的编码
			out = cv2.VideoWriter(os.path.join(SAVE_VIDEO_DIR, self.save_video_name), fourcc, 20.0, (900, 700))
		index = 0
		while self.play and self.IMAGE_HANDLE:
			sleep(1 / 13)
			index += 1
			show = self.IMAGE_HANDLE.read()
			if show is None:
				# 程序执行结束要重置PLC
				try:
					self.plchandle.reset()
				except:
					# print("plc is not in use")
					logger("PLC连接失败", 'error')
					break
				if self.save_video:
					self.update_savevideo.emit(save_video_name)
				break

			rows, cols, channels = show.shape
			if rows != IMG_HEIGHT or cols != IMG_WIDTH:
				show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
			else:
				show = show

			# dest = self.detectorhandle.compute_img(show) if self.work and not self.detectorhandle.finish_job else show
			dest = self.detectorhandle.compute_img(show, index)

			if self.save_video:
				if out is None:
					fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 保存视频的编码
					out = cv2.VideoWriter(os.path.join(SAVE_VIDEO_DIR, self.save_video_name), fourcc, 20.0, (900, 700))
				out.write(dest)

			dest = cv2.cvtColor(dest, cv2.COLOR_BGR2RGB)

			finalimg = QImage(dest.data, dest.shape[1], dest.shape[0], QImage.Format_RGB888)
			self.video_player.setPixmap(QPixmap.fromImage(finalimg))
			self.video_player.setScaledContents(True)

		if self.save_video:
			self.update_savevideo.emit(self.save_video_name)
		# 程序执行结束要重置PLC
		try:
			self.plchandle.reset()
		except:
			logger("PLC重置失败", 'error')

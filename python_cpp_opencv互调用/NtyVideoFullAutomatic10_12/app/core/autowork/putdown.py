import cv2
import numpy as np

from color_test import BaseDetector


class PutdownPosition(object):
	def __init__(self, level=None, row=None, col=None, bag_d=140, left_edge=90, right_edge=510, top_edge=100,
	             bottom_edge=400):
		self.level, self.row, self.col = level, row, col
		self.bag_d = bag_d
		self._cent_x, self._cent_y = None, None
		self._left_edge, self._right_edge, self._top_edge, self._bottom_edge = left_edge, right_edge, top_edge, bottom_edge

	@property
	def cent(self):
		self._cent_x = self._left_edge + self.col * self.bag_d + 0.5 * self.bag_d
		self._cent_y = self._top_edge + self.row * self.bag_d + 0.5 * self.bag_d
		return self._cent_x, self._cent_y

	@property
	def cent_x(self):
		return self._cent_x

	@property
	def cent_y(self):
		return self._cent_y


class PositionUtil(object):
	def __init__(self, init_left_edge=90, init_right_edge=510, init_top_edge=100, init_bottom_edge=400, bag_dim=140):

		self.history = []

		self.left_edge, self.right_edge, self.top_edge, self.bottom_edge, self.bag_dim = init_left_edge, init_right_edge, init_top_edge, \
		                                                                                 init_bottom_edge, bag_dim

		self.positions = []
		self.current_level = 0

	def __getitem__(self, item):
		return self.history[item]

	def fetchimage_cal_edge(self):
		# 实时从摄像头获取放置区域图像
		# TODO left->right  top->bottom
		bagimg = cv2.imread("D:/bags.png")
		gray = cv2.cvtColor(bagimg, cv2.COLOR_BGR2GRAY)
		ret, white_binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

		red_binary, red_contours = BaseDetector().red_contours(bagimg, middle_start=0, middle_end=1300)

		if red_contours is None or len(red_contours) == 0:
			return 100, 530, 110, 390

		x_small, y_small, x_max, y_max = 10000, 10000, 0, 0
		for c in red_contours:
			x, y, w, h = cv2.boundingRect(c)
			if x < x_small: x_small = x
			if y < y_small: y_small = y
			if x_max < x: x_max = x
			if y_max < y: y_max = y

		# print((x_small,y_small),(x_max, y_max))

		# cv2.imshow("bags", red_binary)
		#
		# cv2.waitKey(0)

		return x_small, x_max, y_small, y_max

	def compute_position(self, level=0):
		print("level:{}".format(level))
		if level == 0:
			# history 对应位置设置为1，标名该位置已经放置了袋子
			self.history.append(np.zeros(
				(divmod(self.bottom_edge - self.top_edge, self.bag_dim)[0],
				 divmod(self.right_edge - self.left_edge, self.bag_dim)[0]
				 ), dtype=int))

		else:
			# 重新计算  左右边界  上下边界 并且重新加入history记录情况
			left, right, top, bottom = self.fetchimage_cal_edge()
			self.history.append(np.zeros(
				(divmod(bottom - top, self.bag_dim)[0],
				 divmod(right - left, self.bag_dim)[0]
				 )))

	def add_putdown_positon(self, level=0, row=0, col=0):
		# TODO 真正操作行车将袋子放置在这里
		newposition = PutdownPosition(level=level, row=row, col=col)
		# TODO 一系列的放置动作
		print("real put to ({},{})".format(*newposition.cent))

	def add_bag(self):
		# 添加袋子
		col_item, find, positionarea, row_item = self.find_empty_position()

		if find == True:
			print("level {} th,find a position to put bag({},{})".format(self.current_level, row_item, col_item))

			self.add_putdown_positon(self.current_level, row_item, col_item)
			positionarea[row_item][col_item] = 1
			print("has put to ({},{})".format(row_item, col_item))
		else:
			print("该{}层已满，放置在{}层上".format(self.current_level, self.current_level + 1))
			self.current_level += 1
			col_item, now_find, positionarea, row_item = self.find_empty_position()
			if now_find == False: return
			print("level {} th,find a position to put bag({},{})".format(self.current_level, row_item, col_item))
			self.add_putdown_positon(self.current_level, row_item, col_item)
			positionarea[row_item][col_item] = 1
			print("has put to ({},{})".format(row_item, col_item))

	def find_empty_position(self):
		try:
			positionarea = self.history[self.current_level]
		except Exception as e:
			self.compute_position(self.current_level)
			positionarea = self.history[self.current_level]
		row, col = positionarea.shape
		print(positionarea)
		find = False
		for row_item in range(row):
			for col_item in range(col):
				if positionarea[row_item][col_item] == 0:
					find = True
					break
			if find == True: break
		return col_item, find, positionarea, row_item


if __name__ == "__main__":
	# print(divmod(170, 140))
	# print(divmod(280, 140))
	positionvector = PositionUtil()
	# positionvector.compute_position(level=0)
	# positionvector.compute_position(level=1)
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()
	positionvector.add_bag()

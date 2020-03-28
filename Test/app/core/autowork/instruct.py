from PyQt5.QtCore import QMutex

from app.core.target_detect.pointlocation import PointLocationService


class InstructSender():
	lock = QMutex()

	'''
	
	代码习惯约定：get_moveposition_hock 根据最后的下划线知晓，该方法被钩子线程调用
	
	# 定位钩子
	# 
	# 定位离钩子最近的袋子，并且更新
	# 最近袋子的位置，钩子的位置
	# 
	# 如果钩子与选定袋子距离大于阈值
	# 
	# 移动钩子
	# 
	# 检测钩子与选定袋子的距离，并更新该袋子与钩子的实时位置
	# 
	# 直到钩子与选定袋子的距离小于阈值，则停止移动钩子
	# 
	# 降落钩子
	# 
	# 检测是否拉起
	# 
	# 拉起钩子
	# 
	# 前Y轴负方向移动钩子，直到检测到
	# 卸货区，模拟环境中钩子碰到一条横线即可
	# 
	# 一次循环结束
	# 
	# 钩子复位
	'''

	def __init__(self):
		self.bags = []  # 所有的袋子
		self.hock = None  # 当前的钩子
		self.nearest_bag = None  # 最近的袋子
		self.positionservice = PointLocationService()
		self.movex = None
		self.movey = None
		self._img = None

	@property
	def img(self):
		return self._img

	def location(self):
		# 智能识别线程会调用，并停止钩子移动线程运行
		self.lock.lock()
		# 计算出钩子、最近的袋子、以及虚拟真实距离对比
		img_distance, real_distance, real_x_distance, real_y_distance = self.positionservice.computelocations()
		self.nearest_bag = self.positionservice.nearestbag
		self.hock = self.positionservice.hock
		# 运行定位逻辑,并实时更新钩子与选中袋子实时位置
		print("向PLC中写入需要移动的X、Y轴移动距离")
		self.movex = real_x_distance
		self.movey = real_y_distance
		print("X轴移动：{}，Y轴移动{}".format(real_x_distance, real_y_distance))
		self.img = self.positionservice.img
		self.lock.unlock()

	def move(self):
		# 移动钩子的同时，应该停止图像识别线程
		self.lock.lock()
		print("从PLC中读取需要移动的X、Y轴移动距离,钩子正在移动中。。。。。。")
		print("向X移动{},向Y移动{}".format(self.movex, self.movey))
		self.lock.unlock()

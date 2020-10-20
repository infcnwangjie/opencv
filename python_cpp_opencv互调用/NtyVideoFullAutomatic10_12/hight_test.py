# -*- coding: utf-8 -*-
import random
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

# from app.log.logtool import logger
MOVE_SPEED = 0x032A  # 810行车东西速度

EAST_PLC = 0x0FA0  # 4000    东写入地址
WEST_PLC = 0x0FA1  # 4001    西写入地址
SOUTH_PLC = 0x0FA2  # 4002   南写入地址
NORTH_PLC = 0x0FA3  # 4003   北写入地址
UP_PLC = 0x0FA4  # 4004      上写入地址
DOWN_PLC = 0x0FA5  # 4005    下写入地址

HOCK_MOVE_STATUS_PLC = 0x0FA6  # 4006  行车移动状态写入地址  1：运动  0:静止
HOCK_STOP_PLC = 0x0FA7  # 4007   强制停止写入地址  1:停止  0: 取消限制
HOCK_RESET_PLC = 0x0FA8  # 4008   行车复位写入地址  1 复位 0：取消复位
POWER_PLC = 0x0FA9  # 4009   启动行车  1 启动  0：关闭
GO_AND_BACK = 0x0FAA  # 4010  往返 1 开始 0：结束

UP_CARGOHOOK_PLC = 0x0FAB  # 4011      货钩上写入地址
DOWN_CARGOHOOK_PLC = 0x0FAC  # 4012    货钩下写入地址

SOUTH_NORTH_SERVER_ERROR = 0x1004  # 4100
EAST_SERVER_ERROR = 0x1005  # 4101
WEST_SERVER_ERROR = 0x1006  # 4102
WEST_LIMIT_ERROR = 0x1007  # 4103
EAST_LIMIT_ERROR = 0x1008  # 4104
NORTH_LIMIT_ERROR = 0x1009  # 4105
SOUTH_LIMIT_ERROR = 0x100A  # 4106

HIGH_GIVER                       = 0x1068  # 4200     激光测距仪


class PlcHandle(object):
	'''
	东西南北上下
	紧急停止
	复位 寄存器清零
	'''
	instance = None

	def __init__(self, plc_port="COM7", timeout=0.3):
		self.port = plc_port
		self.timeout = timeout
		self._plc_status = False
		self._power = False
		self.init_plc()

	def __new__(cls, *args, **kwargs):
		if cls.instance == None:
			cls.instance = super().__new__(cls)
		return cls.instance

	def change_port(self, port):
		'''
			切换PLC端口
			:return:
		'''
		try:
			# self.logger = modbus_tk.utils.create_logger("console")
			self.master = modbus_rtu.RtuMaster(
				serial.Serial(port=port, baudrate=19200, bytesize=8, parity='E', stopbits=1, xonxoff=0))
			self.master.set_timeout(self.timeout)  # PLC 延迟
			self.master.set_verbose(True)
			# self.logger.info("connected")
			self._plc_status = True
		except Exception as exc:
			self._plc_status = False
		# self.logger.error("%s- Code=%d", exc, exc.get_exception_code())
		return self._plc_status

	def init_plc(self):
		'''
		初始化PLC
		:return:
		'''
		try:
			self.logger = modbus_tk.utils.create_logger("console")
			self.master = modbus_rtu.RtuMaster(
				serial.Serial(port=self.port, baudrate=19200, bytesize=8, parity='E', stopbits=1, xonxoff=0))
			self.master.set_timeout(self.timeout)  # PLC 延迟
			self.master.set_verbose(True)
			self.logger.info("connected")
			self._plc_status = True
		except Exception as exc:

			for port_index in range(0, 10):
				port = "COM{}".format(port_index)
				# print("test COM{}".format(port_index))
				success = self.change_port(port)
				if success:
					self.port = port
					self._plc_status = True
					break
			else:
				self._plc_status = False
				# logger("%s- Code=%d", exc, exc.get_exception_code())

	@property
	def power(self):
		'''
		1:开启
		0：关闭
		:return:
		'''
		try:
			powervalue = self.__read(POWER_PLC)
		except Exception as e:
			print(e.__str__())

		else:
			self._power = (powervalue == 1)
		return self._power

	@power.setter
	def power(self, trueorfalse):
		'''
				1:开启
				0：关闭
				:return:
				'''
		try:
			self._power = trueorfalse
			self.__write(POWER_PLC, 1 if trueorfalse else 0)
		except Exception as e:
			print(e.__str__())

	def __read(self, address):
		'''
		向PLC中读取数据
		:param address:
		:return:
		'''
		info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=address,
		                           quantity_of_x=1)
		print(info)
		result = info[0]
		return result

	def __write(self, address, value: int):
		'''
		向PLC中写入数据
		:param address:
		:param value:
		:return:
		'''
		if not hasattr(self, 'master') or self.master is None:
			print("PLC 无法写入，请检查端口")
		self.master.execute(1, cst.WRITE_SINGLE_REGISTER, address, output_value=value)

	def get_high(self):
		HIGH_VALUE = self.__read(HIGH_GIVER) #毫米
		print(HIGH_VALUE)
		if HIGH_VALUE is None or HIGH_VALUE =='':
			return 0
		else:
			HIGH_VALUE=float(HIGH_VALUE)
			return HIGH_VALUE

	def is_open(self):
		'''
		用来检测程序是否连接了PLC
		:return:
		'''
		try:
			self.info()
			self._plc_status = True
		except Exception as e:
			print("PLC 连接失败，请检查端口")
			self._plc_status = False

		return self._plc_status

	def init_plc(self):
		'''
		初始化PLC
		:return:
		'''
		try:
			self.master = modbus_rtu.RtuMaster(
				serial.Serial(port=self.port, baudrate=19200, bytesize=8, parity='E', stopbits=1, xonxoff=0))
			self.master.set_timeout(self.timeout)  # PLC 延迟
			self.master.set_verbose(True)
		except Exception as exc:
			print("PLC 连接失败，请检查端口")
			self._plc_status = False

	def read_status(self):
		'''
		读取钩子移动状态： 1：运动   0:静止
		:return: int
		'''
		try:
			result = self.__read(HOCK_MOVE_STATUS_PLC)
		except modbus_tk.modbus.ModbusError as exc:
			print("PLC 无法读取数值，请检查端口")
		return result

	def go_and_back(self):
		try:
			self.__write(GO_AND_BACK, int(1))  # 钩子运动状态上升
		except Exception as exc:
			print("PLC 无法写入数值，请检查端口")

	# ------------------------------------------------
	# 名称：move
	# 功能：写入钩子当前位置
	# 状态：在用
	# 参数： [east,west,south,nourth,up,down,up_cargohook,down_cargohook]   --- 东西南北，定位钩升降，货钩升降
	# 返回： [None]   ---
	# 作者：王杰  2020-5-xx  update 2020-7-06
	# ------------------------------------------------
	def move(self, east=0, west=0, south=0, nourth=0, up=0, down=0, up_cargohook=0, down_cargohook=0):
		try:
			self.__write(HOCK_MOVE_STATUS_PLC, int(1))  # 钩子运动状态上升

			if east != 0:
				self.__write(EAST_PLC, int(east))
			if west != 0:
				self.__write(WEST_PLC, int(west))

			if south != 0:
				self.__write(SOUTH_PLC, int(south))

			if nourth != 0:
				self.__write(NORTH_PLC, int(nourth))

			# 定位钩上升
			if up != 0:
				self.__write(UP_PLC, int(up))
			# 定位钩下降
			if down != 0:
				self.__write(DOWN_PLC, int(down))

			# 	货钩上升
			if up_cargohook != 0:
				self.__write(UP_CARGOHOOK_PLC, int(up_cargohook))

			# 货钩下降
			if down_cargohook != 0:
				self.__write(DOWN_CARGOHOOK_PLC, int(down_cargohook))

		except Exception as exc:
			print("PLC 无法写入数值，请检查端口")

	def hock_stop(self):
		try:
			self.__write(HOCK_MOVE_STATUS_PLC, int(0))
		except Exception as e:
			print("PLC 无法写入数值，请检查端口")

	def ugent_stop(self):
		'''
		紧急停止命令
		'''
		# self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_STOP_PLC, output_value=1)  # 写入
		self.__write(HOCK_STOP_PLC, 1)
		# 运动状态清零
		self.__write(HOCK_MOVE_STATUS_PLC, 0)
		self.__write(EAST_PLC, 0)
		self.__write(WEST_PLC, 0)
		self.__write(SOUTH_PLC, 0)
		self.__write(NORTH_PLC, 0)
		self.__write(UP_PLC, 0)
		self.__write(DOWN_PLC, 0)
		self.__write(UP_CARGOHOOK_PLC, 0)
		self.__write(DOWN_CARGOHOOK_PLC, 0)
		self.__write(HOCK_RESET_PLC, 0)
		self.power = False

	def is_ugent_stop(self):
		result = self.__read(HOCK_STOP_PLC)
		return result

	def reset(self):
		'''复位 1:复位 0：取消复位 '''
		try:
			# 写入复位
			self.__write(HOCK_RESET_PLC, 1)
			# 紧急停止清零
			self.__write(HOCK_STOP_PLC, 0)
			# 运动状态清零
			self.__write(HOCK_MOVE_STATUS_PLC, 0)
			self.__write(EAST_PLC, 0)
			self.__write(WEST_PLC, 0)
			self.__write(SOUTH_PLC, 0)
			self.__write(NORTH_PLC, 0)
			self.__write(UP_PLC, 0)
			self.__write(DOWN_PLC, 0)
			self.__write(UP_CARGOHOOK_PLC, 0)
			self.__write(DOWN_CARGOHOOK_PLC, 0)
			self.power = True

		except Exception as e:
			print("PLC 无法写入数值，请检查端口")

	def info(self):
		# SOUTH_NORTH_SERVER_ERROR = 0x1004  # 4100
		# EAST_SERVER_ERROR = 0x1005  # 4101
		# WEST_SERVER_ERROR = 0x1006  # 4102
		# WEST_LIMIT_ERROR = 0x1007  # 4103
		# EAST_LIMIT_ERROR = 0x1008  # 4104
		# NORTH_LIMIT_ERROR = 0x1009  # 4105
		# SOUTH_LIMIT_ERROR = 0x100A  # 4106
		info = "E:{},W:{},N:{},S:{},UP:{},DOWN:{},POWER:{}，speed:{},SOUTH_NORTH_SERVER_ERROR:{}".format(
			self.__read(EAST_PLC), self.__read(WEST_PLC),
			self.__read(NORTH_PLC), self.__read(SOUTH_PLC),
			self.__read(UP_PLC), self.__read(DOWN_PLC),
			self.__read(POWER_PLC), self.__read(MOVE_SPEED)
			)
		print(info)


if __name__ == '__main__':
	plc = PlcHandle(plc_port='COM7')
	# plc.reset()
	# print(plc.is_open())

	# print(plc.write_error([-1,-2,0]))
	# plc.move(east=2, nourth=3)
	plc.power = True
	# plc.info()
	print(plc.get_high())

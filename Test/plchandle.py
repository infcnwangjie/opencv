# -*- coding: utf-8 -*-
import random
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

EAST_PLC = 0x0FA0  # 4000    东写入地址
WEST_PLC = 0x0FA1  # 4001    西写入地址
SOUTH_PLC = 0x0FA2  # 4002   南写入地址
NORTH_PLC = 0x0FA3  # 4003   北写入地址
UP_PLC = 0x0FA4  # 4004      上写入地址
DOWN_PLC = 0x0FA5  # 4005    下写入地址

HOCK_MOVE_STATUS_PLC = 0x0FA6  # 4006  行车移动状态写入地址  1：运动  0:静止
HOCK_STOP_PLC = 0x0FA7  # 4007   强制停止写入地址  1:停止  0: 取消限制
HOCK_RESET_PLC = 0x0FA8  # 4008   行车复位写入地址  1 复位 0：取消复位


class PlcHandle:
	'''
	东西南北上下
	紧急停止
	复位 寄存器清零
	'''

	def __init__(self, plc_port='COM7', timeout=0.3):
		self.port = plc_port
		self.timeout = timeout
		self._plc_status = False
		self.init_plc()

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
		except modbus_tk.modbus.ModbusError as exc:
			self._plc_status = False
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def __read(self, address):
		'''
		向PLC中读取数据
		:param address:
		:return:
		'''
		info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=address,
		                           quantity_of_x=1)
		result = info[0]
		return result

	def __write(self, address, value: int):
		'''
		向PLC中写入数据
		:param address:
		:param value:
		:return:
		'''
		self.master.execute(1, cst.WRITE_SINGLE_REGISTER, address, output_value=value)

	def is_open(self):
		'''
		用来检测程序是否连接了PLC
		:return:
		'''
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
			self._plc_status = False

	def read_status(self):
		'''
		读取钩子移动状态： 1：运动   0:静止
		:return: int
		'''
		try:
			result = self.__read(HOCK_MOVE_STATUS_PLC)
		# info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=HOCK_MOVE_STATUS_PLC,
		#                            quantity_of_x=1)
		# status_value = info[0]
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())
		return result

	def move(self, east=0, west=0, south=0, nourth=0, up=0, down=0):
		'''
		写入钩子当前位置
		:param position: dict
		:return:
		'''
		try:
			if east != 0:
				self.__write(EAST_PLC, int(east))
			if west != 0:
				self.__write(WEST_PLC, int(west))

			if south != 0:
				self.__write(SOUTH_PLC, int(south))

			if nourth != 0:
				self.__write(NORTH_PLC, int(nourth))

			if up != 0:
				self.__write(UP_PLC, int(up))

			if down != 0:
				self.__write(DOWN_PLC, int(down))

		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def current_position(self):
		east = self.__read(EAST_PLC)
		west = self.__read(WEST_PLC)
		south = self.__read(SOUTH_PLC)
		north = self.__read(NORTH_PLC)
		up = self.__read(UP_PLC)
		down = self.__read(DOWN_PLC)
		return east, west, south, north, up, down

	def ugent_stop(self):
		'''
		紧急停止命令
		'''
		# self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_STOP_PLC, output_value=1)  # 写入
		self.__write(HOCK_STOP_PLC, 1)

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

		except Exception as e:
			pass


if __name__ == '__main__':
	plc = PlcHandle()
	# plc.reset()
	print(plc.read_status())

	# print(plc.write_error([-1,-2,0]))
	plc.move(east=2, nourth=3)
	print("东:{} 西:{} 南:{} 北:{} 上:{} 下:{}".format(*plc.current_position()))
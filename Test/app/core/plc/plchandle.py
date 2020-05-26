# -*- coding: utf-8 -*-
import random
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

from app.config import HOCK_MOVE_STATUS_PLC, \
	HOCK_STOP_PLC, PLC_OPEN, HOCK_CURRENT_X_PLC, HOCK_CURRENT_Y_PLC, HOCK_CURRENT_Z_PLC, TARGET_X_PLC, TARGET_Y_PLC, \
	TARGET_Z_PLC, HOCK_RESET_PLC


class PlcHandle:
	instance = None
	'''
	开始  并向PLC中先写入一个位置
	
	读取　PLC状态，判断PLC是移动还是静止
	
	写入 位置数据
	
	紧急停止
	
	复位 寄存器清零
	
	'''

	def __new__(cls, *args, **kwargs):
		if cls.instance == None:
			cls.instance = super().__new__(cls, *args, **kwargs)
		return cls.instance

	def __init__(self, plc_port='COM3', timeout=5.0):
		self.port = plc_port
		self.timeout = timeout
		if PLC_OPEN:
			try:
				self.logger = modbus_tk.utils.create_logger("console")
				self.init_plc()
				self.logger.info("connected")
				self._plc_status = True
			except:
				self._plc_status = False
		else:
			self._plc_status = False

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
		except modbus_tk.modbus.ModbusError as exc:
			self._plc_status = False
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def read(self, address):
		'''
		向PLC中读取数据
		:param address:
		:return:
		'''
		info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=address,
		                           quantity_of_x=1)
		result = info[0]
		return result

	def write(self, address, value: int):
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
		except modbus_tk.modbus.ModbusError as exc:
			self._plc_status = False
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def read_status(self):
		'''
		读取钩子移动状态： 1：运动   0:静止
		:return: int
		'''
		try:
			result = self.read(HOCK_MOVE_STATUS_PLC)
		# info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=HOCK_MOVE_STATUS_PLC,
		#                            quantity_of_x=1)
		# status_value = info[0]
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())
		return result

	def write_hock_position(self, position):
		'''
		写入钩子当前位置
		:param position: dict
		:return:
		'''
		print("now,write to plc,position is x:{},y:{},z:{}".format(*position))
		if PLC_OPEN:
			try:
				x, y, z = position
				self.write(HOCK_CURRENT_X_PLC, int(x))
				self.write(HOCK_CURRENT_Y_PLC, int(y))
				self.write(HOCK_CURRENT_Z_PLC, int(z))
			except modbus_tk.modbus.ModbusError as exc:
				self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def write_target_position(self, position):
		'''
		写入目标位置
		:param position: dict
		:return:
		'''
		print("now,write to plc,position is x:{},y:{},z:{}".format(*position))
		if PLC_OPEN:
			try:
				x, y, z = position
				self.write(TARGET_X_PLC, int(x))
				self.write(TARGET_Y_PLC, int(y))
				self.write(TARGET_Z_PLC, int(z))
			# self.logger.info(
			# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_X_PLC, output_value=int(x)))  # 写入
			# self.logger.info(
			# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Y_PLC, output_value=int(y)))  # 写入
			# self.logger.info(
			# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Z_PLC, output_value=int(z)))  # 写入
			except modbus_tk.modbus.ModbusError as exc:
				self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def ugent_stop(self):
		'''
		紧急停止命令
		'''
		# self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_STOP_PLC, output_value=1)  # 写入
		self.write(HOCK_STOP_PLC, 1)

	def reset(self):
		'''复位 1:复位 0：取消复位 '''
		print("imgdetector send stop instruct")
		if PLC_OPEN:
			try:
				self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_RESET_PLC, output_value=1)
				# 复位后，写入PLC中的所有数据应该都清除
				self.write_target_position([0, 0, 0])
				# 紧急停止也要清零
				self.write(HOCK_STOP_PLC, 0)
				# 目标地址清零
				self.write_hock_position([0, 0, 0])
			except Exception as e:
				pass

	def __del__(self):
		self.reset()


if __name__ == '__main__':
	plc = PlcHandle()
	print(plc.read_status())
	# plc.ugent_stop()
	# print("plc连接状态：{}".format(plc.is_open()))
	# plc.reset()
	plc.write(HOCK_MOVE_STATUS_PLC, 0)
	print(plc.read_status())
	print(plc.read(HOCK_STOP_PLC))

# print(hex(4004))

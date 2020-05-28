# -*- coding: utf-8 -*-
import random
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

from app.config import HOCK_MOVE_STATUS_PLC, \
	HOCK_STOP_PLC, HOCK_CURRENT_X_PLC, HOCK_CURRENT_Y_PLC, HOCK_CURRENT_Z_PLC, TARGET_X_PLC, TARGET_Y_PLC, \
	TARGET_Z_PLC, HOCK_RESET_PLC


class PlcHandle:
	'''
	开始  并向PLC中先写入一个位置
	
	读取　PLC状态，判断PLC是移动还是静止
	
	写入 位置数据
	
	紧急停止
	
	复位 寄存器清零
	
	'''

	def __init__(self, plc_port='COM3', timeout=0.3):
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

	def write_hock_position(self, position):
		'''
		写入钩子当前位置
		:param position: dict
		:return:
		'''
		print("now,write to plc,position is x:{},y:{},z:{}".format(*position))

		try:
			x, y, z = position
			self.__write(HOCK_CURRENT_X_PLC, int(x))
			self.__write(HOCK_CURRENT_Y_PLC, int(y))
			self.__write(HOCK_CURRENT_Z_PLC, int(z))
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def write_target_position(self, position):
		'''
		写入目标位置
		:param position: dict
		:return:
		'''
		print("now,write to plc,position is x:{},y:{},z:{}".format(*position))
		try:
			x, y, z = position
			self.__write(TARGET_X_PLC, int(x))
			self.__write(TARGET_Y_PLC, int(y))
			self.__write(TARGET_Z_PLC, int(z))
		# self.logger.info(
		# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_X_PLC, output_value=int(x)))  # 写入
		# self.logger.info(
		# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Y_PLC, output_value=int(y)))  # 写入
		# self.logger.info(
		# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Z_PLC, output_value=int(z)))  # 写入
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def current_hock_position(self):
		x = self.__read(HOCK_CURRENT_X_PLC)
		y = self.__read(HOCK_CURRENT_Y_PLC)
		z = self.__read(HOCK_CURRENT_Z_PLC)
		return [x, y, z]

	def target_position(self):
		x = self.__read(TARGET_X_PLC)
		y = self.__read(TARGET_Y_PLC)
		z = self.__read(TARGET_Z_PLC)
		return [x, y, z]

	# def write_error(self, position):
	# 	'''
	# 			写入纠偏量
	# 			:param position: dict
	# 			:return:
	# 			'''
	# 	print("now,write to plc,position is x:{},y:{},z:{}".format(*position))
	# 	try:
	# 		x, y, z = position
	# 		self.__write(ERROR_X_PLC, int(x))
	# 		self.__write(ERROR_Y_PLC, int(y))
	# 		self.__write(ERROR_Z_PLC, int(z))
	# 	# self.logger.info(
	# 	# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_X_PLC, output_value=int(x)))  # 写入
	# 	# self.logger.info(
	# 	# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Y_PLC, output_value=int(y)))  # 写入
	# 	# self.logger.info(
	# 	# 	self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Z_PLC, output_value=int(z)))  # 写入
	# 	except modbus_tk.modbus.ModbusError as exc:
	# 		self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	# def error_position(self):
	# 	'''
	# 	获取纠偏量
	# 	:return:
	# 	'''
	# 	x = self.__read(ERROR_X_PLC)
	# 	y = self.__read(ERROR_Y_PLC)
	# 	z = self.__read(ERROR_Z_PLC)
	# 	return [x, y, z]

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
		print("imgdetector send stop instruct")
		try:
			self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_RESET_PLC, output_value=1)
			# 复位后，写入PLC中的所有数据应该都清除
			self.write_target_position([0, 0, 0])
			# 紧急停止也要清零
			self.__write(HOCK_STOP_PLC, 0)
			# 运动状态清零
			self.__write(HOCK_MOVE_STATUS_PLC, 0)
			# 钩子地址清零
			self.write_hock_position([0, 0, 0])
			# 目标地址清零
			self.write_target_position([0, 0, 0])
		except Exception as e:
			pass

	def __del__(self):
		self.reset()


if __name__ == '__main__':
	plc = PlcHandle()
	# plc.reset()
	# print(plc.read_status())

	# print(plc.write_error([-1,-2,0]))
	# print(plc.target_position())
	# print(plc.current_hock_position())
	# print(plc.error_position())
	print(plc.is_ugent_stop())

# print(hex(4004))

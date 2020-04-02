import random
import serial
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

from app.config import HOCK_MOVE_X_PLC, HOCK_MOVE_Y_PLC, HOCK_MOVE_Z_PLC, HOCK_MOVE_STATUS_PLC, HOCK_STATUS_OPEN, \
	HOCK_STOP_PLC


class PlcHandle:

	def __init__(self, plc_port='COM3', timeout=5.0):
		self.port = plc_port
		self.logger = modbus_tk.utils.create_logger("console")
		self.timeout = timeout
		self.init_plc()
		self.logger.info("connected")
		self._plc_status = True

	@property
	def status(self):
		return self._plc_status

	def init_plc(self):
		try:
			self.master = modbus_rtu.RtuMaster(
				serial.Serial(port=self.port, baudrate=19200, bytesize=8, parity='E', stopbits=1, xonxoff=0))
			self.master.set_timeout(self.timeout)  # PLC 延迟
			self.master.set_verbose(True)
		except modbus_tk.modbus.ModbusError as exc:
			self._plc_status = False
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())


	def write_position(self, position):
		print("now,write to plc,position is x:{},y:{},z:{}".format(*position))
		try:
			x, y, z = position
			self.logger.info(
				self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_X_PLC, output_value=int(x)))  # 写入
			self.logger.info(
				self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Y_PLC, output_value=int(y)))  # 写入
			self.logger.info(
				self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_MOVE_Z_PLC, output_value=int(z)))  # 写入
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())

	def read_status(self):
		# 可能有 ready、move
		try:
			info = self.master.execute(1, cst.READ_HOLDING_REGISTERS, starting_address=HOCK_MOVE_STATUS_PLC,
			                           quantity_of_x=1)
			status_value = info[0]
		except modbus_tk.modbus.ModbusError as exc:
			self.logger.error("%s- Code=%d", exc, exc.get_exception_code())
		return status_value

	def ugent_stop(self):
		'''紧急停止命令'''
		print("imgdetector send stop instruct")
		self.logger.info(
			self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_STOP_PLC, output_value=1))  # 写入

	def reset(self):
		'''复位'''
		print("imgdetector send stop instruct")
		self.logger.info(
			self.master.execute(1, cst.WRITE_SINGLE_REGISTER, HOCK_STOP_PLC, output_value=0))  # 写入

	def __del__(self):
		self.reset()


if __name__ == '__main__':
	plc = PlcHandle()
	print(plc.read_status())
	plc.ugent_stop()
	print(plc.status)

# print(hex(4004))

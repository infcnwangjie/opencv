import random


class PlcHandle:

	def write_position(self, position):
		print("write to plc")
		print(position)

	def read_status(self):
		print("读取状态")
		# 可能有 ready、move
		statuslist = ['ready_without_p', 'ready_with_p', 'move', 'finish']
		return random.choice(statuslist)


if __name__ == '__main__':
	plc = PlcHandle()
	print(plc.read_status())

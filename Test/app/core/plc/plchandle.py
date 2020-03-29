import random


class PlcHandle:

    def write_position(self, position):
        print("now,write to plc,position is x:{},y:{},z:{}".format(*position))

    def read_status(self):
        # 可能有 ready、move
        statuslist = ['move', 'stop', 'reback', 'finish']
        status = random.choice(statuslist)
        print("hockthread read plc status:{}".format(status))
        return status


if __name__ == '__main__':
    plc = PlcHandle()
    print(plc.read_status())

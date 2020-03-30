import random


class PlcHandle:

    def write_position(self, position):
        print("now,write to plc,position is x:{},y:{},z:{}".format(*position))

    def read_status(self):
        # 可能有 ready、move
        statuslist = ['move', 'stop', 'reback']
        status = random.choice(statuslist)
        print("hockthread read plc status:{}".format(status))
        return status

    def write_stop(self):
        print("imgdetector send stop instruct")


if __name__ == '__main__':
    plc = PlcHandle()
    print(plc.read_status())

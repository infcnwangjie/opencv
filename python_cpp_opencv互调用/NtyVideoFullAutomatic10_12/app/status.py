# encoding:utf-8
class HockStatus:
	POSITION_NEARESTBAG = 0 #开始状态，寻找最近袋子
	DROP_HOCK = 1  # 放钩
	PULL_HOCK = 2  # 拉钩子
	FIND_CONVEYERBELT=3 #找到传送带
	DROP_BAG = 4  # 结束
	REBACK = 5  # 复位

class Landmark_Model_Select:
	CHOOSE_X_SAME=0
	CHOOSE_Y_SAME=1
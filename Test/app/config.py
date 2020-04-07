# encoding:utf-8
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基准目录

ICON_DIR = os.path.join(BASE_DIR, 'icons')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train_data')  # 数字训练

TEMPLATES_PATH = [os.path.join(TRAIN_DATA_DIR, file) for file in os.listdir(TRAIN_DATA_DIR) if
                  re.match("template\d{1,2}.png", file)]

NEG_TEMPLATES_PATH = [os.path.join(TRAIN_DATA_DIR, file) for file in os.listdir(TRAIN_DATA_DIR) if
                  re.match("neg_template\d{1,2}.png", file)]
#
# FIRST_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template1.png')  # 模板图片
#
# SECOND_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template2.png')  # 模板图片

RECT_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'rect_template.png')  # 模板图片

DETECT_BY_MULTIPLEAREA = 0  # 多区域混合计算

DETECT_BY_HSVINRANGE = 1  # 单一hsv阈值区间计算

DISTANCE_LASTER_HOCK_X = 60  # 灯与钩子X轴误差距离60cm

DISTANCE_LASTER_HOCK_Y = 60  # 灯与钩子Y轴误差距离60cm

DISTANCE_LANDMARK_SPACE = 200  # 地标与地标之间间隔大概2米

DEBUG = False
SDK_OPEN = True
PLC_OPEN = False
#####海康威视图像动态连接库#############################################################
MVCAMERACONTROLDLL_PATH = 'C:/Program Files/MVS/MvCameraControl.dll'

####PLC存放变量地址#####################################################################
HOCK_MOVE_X_PLC = 0x0FA0
HOCK_MOVE_Y_PLC = 0x0FA1
HOCK_MOVE_Z_PLC = 0x0FA2
HOCK_STOP_PLC = 0x0FA7
HOCK_CURRENT_X_PLC = 0x0FA3
HOCK_CURRENT_Y_PLC = 0x0FA4
HOCK_CURRENT_Z_PLC = 0x0FA5
HOCK_MOVE_STATUS_PLC = 0x0FA6

#####PLC存放钩子状态编码################################################################
HOCK_STATUS_OPEN = 1
HOCK_STATUS_OFF = 2
HOCK_STATUS_READY = 3
HOCK_STATUS_MOVE = 4

# print(NEG_TEMPLATES_PATH)

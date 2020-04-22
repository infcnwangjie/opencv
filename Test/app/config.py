# encoding:utf-8
import os
import re

from app.log.logtool import mylog_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基准目录

mylog_error(BASE_DIR)

DETECT_BY_MULTIPLEAREA = 0  # 多区域混合计算

DETECT_BY_HSVINRANGE = 1  # 单一hsv阈值区间计算

DISTANCE_LASTER_HOCK_X = 60  # 灯与钩子X轴误差距离60cm

DISTANCE_LASTER_HOCK_Y = 60  # 灯与钩子Y轴误差距离60cm

DEBUG = True
SDK_OPEN = False
PLC_OPEN = False

DISTANCE_SAMEXLANDMARK_SPACE = 200 if not DEBUG else 20  # 地标与地标之间间隔大概2米

DISTANCE_SAMEYLANDMARK_SPACE = 400  # 地标与地标之间间隔大概2米

#######################################################################################
IMG_WIDTH = 900
IMG_HEIGHT = 700

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


VIDEO_DIR='C:/NTY_IMG_PROCESS/VIDEO'
ROIS_DIR='C:/NTY_IMG_PROCESS/ROIS'

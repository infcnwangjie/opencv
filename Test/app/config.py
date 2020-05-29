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

DEBUG = False
SDK_OPEN = True

DISTANCE_SAMEXLANDMARK_SPACE = 200 if not DEBUG else 20  # 地标与地标之间间隔大概2米

DISTANCE_SAMEYLANDMARK_SPACE = 400  # 地标与地标之间间隔大概2米

#######################################################################################
IMG_WIDTH = 900
IMG_HEIGHT = 700
LEFT_MARK_FROM = int(0.168 * IMG_WIDTH)
LEFT_MARK_TO = int(0.195 * IMG_WIDTH)
RIGHT_MARK_FROM = int(0.848 * IMG_WIDTH)
RIGHT_MARK_TO = int(0.876 * IMG_WIDTH)

#####海康威视图像动态连接库#############################################################
MVCAMERACONTROLDLL_PATH = 'C:/Program Files/MVS/MvCameraControl.dll'

####PLC存放变量地址#####################################################################
EAST_PLC = 0x0FA0  # 4000    东写入地址
WEST_PLC = 0x0FA1  # 4001    西写入地址
SOUTH_PLC = 0x0FA2  # 4002   南写入地址
NORTH_PLC = 0x0FA3  # 4003   北写入地址
UP_PLC = 0x0FA4  # 4004      上写入地址
DOWN_PLC = 0x0FA5  # 4005    下写入地址

HOCK_MOVE_STATUS_PLC = 0x0FA6  # 4006  行车移动状态写入地址  1：运动  0:静止
HOCK_STOP_PLC = 0x0FA7  # 4007   强制停止写入地址  1:停止  0: 取消限制
HOCK_RESET_PLC = 0x0FA8  # 4008   行车复位写入地址  1 复位 0：取消复位
# print(NEG_TEMPLATES_PATH)

VIDEO_DIR = 'D:/PIC/MV-CA060-10GC (00674709176)'
ROIS_DIR = 'D:/NTY_IMG_PROCESS/ROIS'
SAVE_VIDEO_DIR='D:/NTY_IMG_PROCESS/VIDEO'
PROGRAM_DATA_DIR = 'D:/NTY_IMG_PROCESS/DATA'
BAGROI_DIR = "D:/NTY_IMG_PROCESS/BAG_ROI"
SUPPORTREFROI_DIR = "D:/NTY_IMG_PROCESS/SUPPORT_ROI"

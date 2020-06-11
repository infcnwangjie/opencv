# encoding:utf-8
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基准目录

DETECT_BY_MULTIPLEAREA = 0  # 多区域混合计算

DETECT_BY_HSVINRANGE = 1  # 单一hsv阈值区间计算

DISTANCE_LASTER_HOCK_X = 60  # 灯与钩子X轴误差距离60cm

DISTANCE_LASTER_HOCK_Y = 60  # 灯与钩子Y轴误差距离60cm

DEBUG = True  #开启测试模式
SDK_OPEN = True #开启海康摄像头

DISTANCE_SAMEXLANDMARK_SPACE = 200 if not DEBUG else 20  # 地标与地标之间间隔大概2米

DISTANCE_SAMEYLANDMARK_SPACE = 400  # 地标与地标之间间隔大概2米

#######################################################################################
IMG_WIDTH = 900  #限定图像宽度
IMG_HEIGHT = 700 #限定图像高度
LEFT_MARK_FROM = int(0.168 * IMG_WIDTH) #左侧部分开始
LEFT_MARK_TO = int(0.195 * IMG_WIDTH) #左侧部分结束
RIGHT_MARK_FROM = int(0.848 * IMG_WIDTH) #右侧部分开始
RIGHT_MARK_TO = int(0.876 * IMG_WIDTH) #右侧部分结束

MVCAMERACONTROLDLL_PATH = 'c:/Program Files/MVS/MvCameraControl.dll' #海康威视图像动态连接库


VIDEO_DIR = 'D:/PIC/MV-CA060-10GC (00674709176)' # 海康视频路径
ROIS_DIR = 'D:/NTY_IMG_PROCESS/ROIS' #海康ROI存储路径
SAVE_VIDEO_DIR='D:/NTY_IMG_PROCESS/VIDEO' #视频留存路径
PROGRAM_DATA_DIR = 'D:/NTY_IMG_PROCESS/DATA' #其他数据存储路径
BAGROI_DIR = "D:/NTY_IMG_PROCESS/BAG_ROI" #袋子ROI存储
HOCK_ROI="D:/NTY_IMG_PROCESS/HOCK_ROI" #钩子ROI
SUPPORTREFROI_DIR = "D:/NTY_IMG_PROCESS/SUPPORT_ROI"

LOG_PATH="D:/NTY_IMG_PROCESS/LOGS"# 日志存储文档

PLC_COM='COM7' #PLC端口号
LASTER_HOCK_DISTANCE=30 #cm
HOCK_DISTANCE=30 #CM

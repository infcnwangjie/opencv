import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基准目录

TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train_data')  # 数字训练

FIRST_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template1.png')  # 模板图片

SECOND_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template2.png')  # 模板图片

THIRD_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template3.png')  # 模板图片

FIRST_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template1.png')  # 模板图片

SECOND_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template2.png')  # 模板图片

RECT_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'rect_template.png')  # 模板图片

DETECT_BY_MULTIPLEAREA = 0  # 多区域混合计算

DETECT_BY_HSVINRANGE = 1  # 单一hsv阈值区间计算

DISTANCE_LASTER_HOCK = 60  # 灯与摄像头距离大概60公分

DISTANCE_LANDMARK_SPACE = 200  # 地标与地标之间间隔大概2米


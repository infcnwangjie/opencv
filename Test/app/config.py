import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #根目录

TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train_data')#训练数据目录

FIRST_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template1.png')

SECOND_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template2.png')

THIRD_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'template3.png')

FIRST_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template1.png')

SECOND_NEG_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'neg_template2.png')

RECT_TEMPLATE_PATH = os.path.join(TRAIN_DATA_DIR, 'rect_template.png')

DETECT_BY_MULTIPLEAREA = 0

DETECT_BY_HSVINRANGE = 1

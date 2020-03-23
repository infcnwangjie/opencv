import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_DIR=os.path.join(BASE_DIR,'train_data')

FIRST_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'template1.png')

SECOND_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'template2.png')

THIRD_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'template3.png')

FIRST_NEG_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'neg_template1.png')

SECOND_NEG_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'neg_template2.png')

RECT_TEMPLATE_PATH=os.path.join(TRAIN_DATA_DIR,'rect_template.png')
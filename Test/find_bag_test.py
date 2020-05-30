# -*- coding: utf-8 -*-
# from gevent import monkey;
import os
import pickle
import random
from collections import defaultdict
from functools import cmp_to_key
from operator import itemgetter

import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, ROIS_DIR, LEFT_MARK_FROM, LEFT_MARK_TO, RIGHT_MARK_FROM, RIGHT_MARK_TO, \
	PROGRAM_DATA_DIR, SUPPORTREFROI_DIR
import cv2
import time
import gevent
import profile
from app.core.beans.models import LandMarkRoi, NearLandMark, TargetRect
from app.core.exceptions.allexception import NotFoundLandMarkException
from app.core.processers.bag_detector import BagDetector
from app.core.processers.preprocess import AbstractDetector
from app.log.logtool import mylog_error
import re

try:
	result=3/0
except Exception as e:
	print(e.__str__())
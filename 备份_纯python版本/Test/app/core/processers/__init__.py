from functools import partial

import cv2

from app.config import IMG_HEIGHT, IMG_WIDTH
from app.core.beans.models import LandMarkRoi, BagRoi
from app.log.logtool import logger
import numpy as np




class SmallWords(type):
	def __new__(cls, name, bases, attrs):
		if attrs is None:
			attrs = {}
		attrs['logger'] = logger
		# attrs['find_it'] = findit
		cls.instance = None

		return super().__new__(cls, name, bases, attrs)

	def __call__(self, *args, **kwargs):
		if self.instance is not None:
			return self.instance
		else:
			return super().__call__(*args, **kwargs)

import cv2
from functools import partial

from app.log.logtool import logger


class SmallWords(type):
	def __new__(cls, name, bases, attrs):
		if attrs is None:
			attrs = {}
		attrs['logger'] = logger
		attrs['find_it'] = partial(cv2.calcBackProject,channels=[0, 1],ranges=[0, 180, 0, 256],scale=1)
		cls.instance = None
		return super().__new__(cls, name, bases, attrs)

	def __call__(self, *args, **kwargs):
		if self.instance is not None:
			return self.instance
		else:
			return super().__call__(*args, **kwargs)
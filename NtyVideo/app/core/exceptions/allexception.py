# -*- coding: utf-8 -*-
class SdkException(Exception):
	def __init__(self, msg):
		super().__init__(msg)

class NotFoundBagException(Exception):
	def __init__(self,msg):
		super().__init__(msg)


class NotFoundHockException(Exception):
	def __init__(self,msg):
		super().__init__(msg)


class NotFoundLandMarkException(Exception):
	'''
	未检测出地标异常
	'''
	def __init__(self,msg):
		super().__init__(msg)

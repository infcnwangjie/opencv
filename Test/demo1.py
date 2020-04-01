import os


import re
from _ctypes import byref

avi_pattern=re.compile("\d{4}-\d{2}-\d{2}-\d{2}.*")

def video():
	videos = []
	for file in os.listdir("D:/video"):
		# if os.path.isfile(file):
		matchresult=re.match(avi_pattern,file)
		# print(file)
		if matchresult:
			print(matchresult.group(0))


# video()
from ctypes import cdll, c_uint, c_void_p, c_int, c_float, c_char_p

MWORKDLL = cdll.LoadLibrary("E:/git/cpp/python_c/libDemo.so")

MWORKDLL.call_a.argtype = (c_int, c_float, c_char_p)
MWORKDLL.call_a.restype = c_int
a = c_int(10)
b = c_float(12.34)
word = (c_char_p)("Fine,thank you")
print(MWORKDLL.call_a(a,b,word))


import logging
import os
import time

from app.config import LOG_PATH
# LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "  # 配置输出日志格式
LOG_FORMAT = "%(asctime)s %(levelname)s  %(message)s "  # 配置输出日志格式
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
log_path = LOG_PATH




today_str = time.strftime("%Y-%m-%d", time.localtime())
filename_path = os.path.join(log_path, today_str + ".log")
try:
	if os.path.exists(log_path):
		pass
	else:
		os.mkdir(log_path)

	if os.path.exists(filename_path):
		pass
	else:
		d=open(filename_path,'w')
		d.write("\n")

except:
	pass



logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT,
                    filename=os.path.join(log_path, today_str + ".log")  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                    )


def mylog_debug(msg):
	logging.debug(msg)

def mylog_info(msg):
	logging.info(msg)

def mylog_warning(msg):
	logging.warning(msg)


def mylog_error(msg):
	logging.error(msg)


def mylog_critical(msg):
	logging.critical(msg)


method_dict = dict(debug=mylog_debug,info=mylog_info, warn=mylog_warning, error=mylog_error, serious=mylog_critical)


def logger( msg, level: str):
	if level in method_dict:
		method = method_dict[level]
	else:
		method = mylog_debug
	# method=mylog_debug
	method(msg)

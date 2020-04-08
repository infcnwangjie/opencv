import logging
import os
import time

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "  # 配置输出日志格式
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
log_path=r"c:\machine_vision_logs"
try:
	if os.path.exists(log_path):
		pass
	else:
		os.mkdir(log_path)
except:
	pass

today_str = time.strftime("%Y-%m-%d", time.localtime())

logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT,
                    filename=os.path.join(log_path,today_str+".log")  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                    )


def mylog_debug(msg):
	logging.debug(msg)


def mylog_warning(msg):
	logging.warning(msg)


def mylog_error(msg):
	logging.error(msg)


def mylog_critical(msg):
	logging.critical(msg)



# mylog_error("wocao")
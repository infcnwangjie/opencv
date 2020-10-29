# encoding:utf-8

config_dic = {}

LANDMARK_COLOR_INFO = {}


def load_landmark_color_info(key, value):
	global LANDMARK_COLOR_INFO
	LANDMARK_COLOR_INFO[key.strip()] = value.strip()


def load_config():
	import re
	config_pattern = re.compile("([A-Za-z_0-9]+)\s*\=\s*(.*)")
	try:
		with open("D:/NTY_IMG_PROCESS/DATA/config.txt", 'rt', encoding="utf-8") as config_handle:
			for line in config_handle.readlines():
				if len(line) == 0 or line == "":
					continue
				match_result = re.match(config_pattern, line)
				if match_result is not None:
					key = match_result.group(1)
					value = match_result.group(2)
					config_dic[key] = value
					if 'COLOR' in key:
						load_landmark_color_info(key, value)

	# print("key:{},value:{}".format(key,value))
	except Exception as e:
		raise e


load_config()
WAIT_TIME = int(config_dic['WAIT_TIME'])
DETECT_BY_MULTIPLEAREA = int(config_dic['DETECT_BY_MULTIPLEAREA'])

DETECT_BY_HSVINRANGE = int(config_dic['DETECT_BY_HSVINRANGE'])

DISTANCE_CAMERA_HOCK = int(config_dic['DISTANCE_CAMERA_HOCK'])

DEBUG = True if 'True' in config_dic['DEBUG'] else False
BIG_OR_SMALL_LASTER = int(config_dic['BIG_OR_SMALL_LASTER'])
IMG_WIDTH = int(config_dic['IMG_WIDTH'])
IMG_HEIGHT = int(config_dic['IMG_HEIGHT'])

land_mark_left_start = int(config_dic['land_mark_left_start'])
land_mark_left_end = int(config_dic['land_mark_left_end'])
land_mark_right_start = int(config_dic['land_mark_right_start'])
land_mark_right_end = int(config_dic['land_mark_right_end'])

middle_start_withlandmark = int(config_dic['middle_start_withlandmark'])
middle_end_withlandmark = int(config_dic['middle_end_withlandmark'])

middle_start_withoutlandmark = int(config_dic['middle_start_withoutlandmark'])
middle_end_withoutlandmark = int(config_dic['middle_end_withoutlandmark'])

bag_real_width = int(config_dic['bag_real_width'])
bag_min_area = int(config_dic['bag_min_area'])
bag_max_area = int(config_dic['bag_max_area'])
permissible_distance = int(config_dic['permissible_distance'])
move_tobag_retry_times = int(config_dic['move_tobag_retry_times'])
hock_y_error = int(config_dic['hock_y_error'])
hock_x_error = int(config_dic['hock_x_error'])
direct_choice = config_dic['direct_choice']

smalllaster_min_width = int(config_dic['smalllaster_min_width'])
smalllaster_min_height = int(config_dic['smalllaster_min_height'])
smalllaster_max_width = int(config_dic['smalllaster_max_width'])
smalllaster_max_height = int(config_dic['smalllaster_max_height'])
smalllaster_min_area = int(config_dic['smalllaster_min_area'])
smalllaster_max_area = int(config_dic['smalllaster_max_area'])


biglaster_min_width = int(config_dic['biglaster_min_width'])
biglaster_min_height = int(config_dic['biglaster_min_height'])
biglaster_max_width = int(config_dic['biglaster_max_width'])
biglaster_max_height = int(config_dic['biglaster_max_height'])
biglaster_min_area = int(config_dic['biglaster_min_area'])
biglaster_max_area = int(config_dic['smalllaster_max_area'])

laster_min_width = int(config_dic['laster_min_width'])
laster_max_width = int(config_dic['laster_max_width'])
laster_min_height = int(config_dic['laster_min_height'])
laster_max_height = int(config_dic['laster_max_height'])
laster_min_area = int(config_dic['laster_min_area'])
laster_max_area = int(config_dic['laster_max_area'])

PLAT = str(config_dic['PLAT'])
MVCAMERACONTROLDLL_PATH = config_dic['MVCAMERACONTROLDLL_PATH']
SUPPLY_OPENCV_DLL_64_PATH = config_dic['SUPPLY_OPENCV_DLL_64_PATH']
SUPPLY_OPENCV_DLL_32_PATH = config_dic['SUPPLY_OPENCV_DLL_32_PATH']

VIDEO_DIR = config_dic['VIDEO_DIR']
ROIS_DIR = config_dic['ROIS_DIR']
SAVE_VIDEO_DIR = config_dic['SAVE_VIDEO_DIR']
PROGRAM_DATA_DIR = config_dic['PROGRAM_DATA_DIR']
BAGROI_DIR = config_dic['BAGROI_DIR']
HOCK_ROI = config_dic['HOCK_ROI']
SUPPORTREFROI_DIR = config_dic['SUPPORTREFROI_DIR']
LOG_PATH = config_dic['LOG_PATH']
PLC_COM = config_dic['PLC_COM']
LASTER_HOCK_DISTANCE = int(config_dic['LASTER_HOCK_DISTANCE'])  # cm
HOCK_DISTANCE = int(config_dic['HOCK_DISTANCE'])  # CM
# COLOR_RANGE = dict(GREEN=[[35, 43, 46], [77, 255, 255]], RED=[[156, 43, 46], [180, 255, 255]],
#                    YELLOW=[[11, 43, 46], [34, 255, 255]], BLUE=[[100, 43, 46], [124, 255, 255]],
#                    ORANGE=[[11, 43, 46], [25, 255, 255]], CYAN=[[78, 43, 46], [99, 255, 255]],
#                    PURPLE=[[125, 43, 46], [155, 255, 255]])

COLOR_RANGE = dict(GREEN=[[35, 43, 46], [77, 255, 255]], RED=[[156, 43, 46], [180, 255, 255]],
                   YELLOW=[[11, 43, 46], [34, 255, 255]], BLUE=[[100, 43, 46], [124, 255, 255]]
                   )

LANDMARK_THREHOLD_START = int(config_dic['LANDMARK_THREHOLD_START'])  # CM
LANDMARK_THREHOLD_END = int(config_dic['LANDMARK_THREHOLD_END'])  # CM


def reload():
	load_config()
	global DETECT_BY_MULTIPLEAREA, DETECT_BY_HSVINRANGE, DISTANCE_CAMERA_HOCK, DEBUG, IMG_WIDTH, IMG_HEIGHT, \
		land_mark_left_start, land_mark_left_end, land_mark_right_start, land_mark_right_end, middle_start_withlandmark, \
		middle_end_withlandmark, middle_start_withoutlandmark, middle_end_withoutlandmark, bag_real_width, \
		bag_min_area, bag_max_area, permissible_distance, move_tobag_retry_times, hock_y_error, direct_choice, \
		hock_min_width, hock_min_height, hock_max_width, hock_max_height, hock_min_area, hock_max_area, \
		PLAT, MVCAMERACONTROLDLL_PATH, SUPPLY_OPENCV_DLL_64_PATH, SUPPLY_OPENCV_DLL_32_PATH, \
		VIDEO_DIR, ROIS_DIR, SAVE_VIDEO_DIR, PROGRAM_DATA_DIR, BAGROI_DIR, HOCK_ROI, SUPPORTREFROI_DIR, \
		LOG_PATH, PLC_COM, LASTER_HOCK_DISTANCE, HOCK_DISTANCE, LANDMARK_THREHOLD_START, LANDMARK_THREHOLD_END, WAIT_TIME, BIG_OR_SMALL_LASTER

	DETECT_BY_MULTIPLEAREA = int(config_dic['DETECT_BY_MULTIPLEAREA'])
	WAIT_TIME = int(config_dic['WAIT_TIME'])
	BIG_OR_SMALL_LASTER = int(config_dic['BIG_OR_SMALL_LASTER'])

	DETECT_BY_HSVINRANGE = int(config_dic['DETECT_BY_HSVINRANGE'])

	DISTANCE_CAMERA_HOCK = int(config_dic['DISTANCE_CAMERA_HOCK'])

	DEBUG = True if 'True' in config_dic['DEBUG'] else False
	IMG_WIDTH = int(config_dic['IMG_WIDTH'])
	IMG_HEIGHT = int(config_dic['IMG_HEIGHT'])

	land_mark_left_start = int(config_dic['land_mark_left_start'])
	land_mark_left_end = int(config_dic['land_mark_left_end'])
	land_mark_right_start = int(config_dic['land_mark_right_start'])
	land_mark_right_end = int(config_dic['land_mark_right_end'])

	middle_start_withlandmark = int(config_dic['middle_start_withlandmark'])
	middle_end_withlandmark = int(config_dic['middle_end_withlandmark'])

	middle_start_withoutlandmark = int(config_dic['middle_start_withoutlandmark'])
	middle_end_withoutlandmark = int(config_dic['middle_end_withoutlandmark'])

	bag_real_width = int(config_dic['bag_real_width'])
	bag_min_area = int(config_dic['bag_min_area'])
	bag_max_area = int(config_dic['bag_max_area'])
	permissible_distance = int(config_dic['permissible_distance'])
	move_tobag_retry_times = int(config_dic['move_tobag_retry_times'])
	hock_y_error = int(config_dic['hock_y_error'])
	hock_x_error = int(config_dic['hock_x_error'])
	direct_choice = config_dic['direct_choice']

	hock_min_width = int(config_dic['hock_min_width'])
	hock_min_height = int(config_dic['hock_min_height'])
	hock_max_width = int(config_dic['hock_max_width'])
	hock_max_height = int(config_dic['hock_max_height'])
	hock_min_area = int(config_dic['hock_min_area'])
	hock_max_area = int(config_dic['hock_max_area'])

	PLAT = str(config_dic['PLAT'])
	MVCAMERACONTROLDLL_PATH = config_dic['MVCAMERACONTROLDLL_PATH']
	SUPPLY_OPENCV_DLL_64_PATH = config_dic['SUPPLY_OPENCV_DLL_64_PATH']
	SUPPLY_OPENCV_DLL_32_PATH = config_dic['SUPPLY_OPENCV_DLL_32_PATH']

	VIDEO_DIR = config_dic['VIDEO_DIR']
	ROIS_DIR = config_dic['ROIS_DIR']
	SAVE_VIDEO_DIR = config_dic['SAVE_VIDEO_DIR']
	PROGRAM_DATA_DIR = config_dic['PROGRAM_DATA_DIR']
	BAGROI_DIR = config_dic['BAGROI_DIR']
	HOCK_ROI = config_dic['HOCK_ROI']
	SUPPORTREFROI_DIR = config_dic['SUPPORTREFROI_DIR']
	LOG_PATH = config_dic['LOG_PATH']
	PLC_COM = config_dic['PLC_COM']
	LASTER_HOCK_DISTANCE = int(config_dic['LASTER_HOCK_DISTANCE'])
	HOCK_DISTANCE = int(config_dic['HOCK_DISTANCE'])
	LANDMARK_THREHOLD_START = int(config_dic['LANDMARK_THREHOLD_START'])
	LANDMARK_THREHOLD_END = int(config_dic['LANDMARK_THREHOLD_END'])
	return True

# print(LANDMARK_COLOR_INFO)

from time import sleep

import cv2
import numpy as np
from app.config import IMG_HEIGHT, IMG_WIDTH, COLOR_RANGE
from app.core.autowork.detector import BagDetector, HockDetector, LandMarkDetecotr, shrink_coach_area
from app.log.logtool import logger


def get_colorrange_binary(color_code=None, target=None, color_low=None, color_high=None):
	hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
	if color_code is not None:
		color_low, color_high = COLOR_RANGE[color_code]
	color1_min, color1_max = np.array(color_low), np.array(color_high)
	foreground = cv2.inRange(hsv, color1_min, color1_max)

	b, g, r = cv2.split(target)
	if color_code == 'GREEN':
		gx_ignore, gy_ignore = np.where((b > 100) | (r > 100))
		g[gx_ignore, gy_ignore] = 0
		ret, g = cv2.threshold(g, 140,
		                       255,
		                       cv2.THRESH_BINARY)  # 110,255
		g = cv2.bitwise_and(g, g, mask=foreground)

		return g

	if color_code == 'RED':
		rx_ignore, ry_ignore = np.where((g > 100) | (b > 100))
		r[rx_ignore, ry_ignore] = 0
		ret, r = cv2.threshold(r, 140,
		                       255,
		                       cv2.THRESH_BINARY)  # 110,255

		r = cv2.bitwise_and(r, r, mask=foreground)

		return r

	if color_code == 'BLUE':
		bx_ignore, by_ignore = np.where((g > 100) | (r > 100))
		b[bx_ignore, by_ignore] = 0
		ret, b = cv2.threshold(b, 140,
		                       255,
		                       cv2.THRESH_BINARY)  # 110,255

		b = cv2.bitwise_and(b, b, mask=foreground)

		return b

	return foreground

def test_landmark_bag():
	import cv2
	import numpy as np
	show = None

	cv2.namedWindow("show")

	# cv2.namedWindow("foreground")
	def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
		global show
		if event == cv2.EVENT_LBUTTONDOWN:
			xy = "% d, % d" % (x, y)
			print('x, y = {},{}'.format(x, y))

	# TODO https://blog.csdn.net/ikerpeng/article/details/47972959

	loc = cv2.setMouseCallback("show", on_EVENT_LBUTTONDOWN)

	cap = cv2.VideoCapture("D:/Video_20201023102602374.avi")  # Video_20200725072828533.avi
	# cap = SdkHandle()
	# cap = cv2.VideoCapture("D:\\Video_20200928150629723.avi")  # Video_20200725072828533.avi
	landmark_detect = LandMarkDetecotr()
	hock_detect = HockDetector()
	bag_detect = BagDetector()

	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			break
		rows, cols, channels = show.shape
		rows, cols, channels = show.shape
		# if rows != IMG_HEIGHT or cols != IMG_WIDTH:
		# 	show = cv2.resize(show, (IMG_HEIGHT, IMG_WIDTH))
		# else:
		# 	show = show
		show = cv2.resize(show, (300, IMG_HEIGHT))

		r=get_colorrange_binary("RED",show
		                      )
		disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		r = cv2.filter2D(r, -1, disc)

		g = get_colorrange_binary("GREEN", show
		                          )
		g = cv2.filter2D(g, -1, disc)
		# cv2.imshow(color_code, foreground)

		b = get_colorrange_binary("BLUE", show
		                          )
		b = cv2.filter2D(b, -1, disc)


		# cv2.imshow("b",b)
		# cv2.imshow("g", g)
		# cv2.imshow("r", r)
		result=np.concatenate((r,g,b), axis=1)
		cv2.imshow("result",result)
		cv2.imshow("show",show)



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


test_landmark_bag()
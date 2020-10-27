from time import sleep





def test_bags():
	# TODO https://blog.csdn.net/ikerpeng/article/details/47972959
	import cv2
	import numpy as np
	cap = cv2.VideoCapture("D:\\Video_20200831092239170.avi")  # Video_20200725072828533.avi

	# cap = cv2.VideoCapture("D:\\Video_20200928150629723.avi")#Video_20200725072828533.avi
	# cap = SdkHandle()

	# cv2.namedWindow("show")
	# cv2.namedWindow("foreground")
	while True:
		sleep(1 / 13)
		ret, show = cap.read()
		if show is None:
			continue
		rows, cols, channels = show.shape
		if rows != 700 or cols != 900:
			show = cv2.resize(show, (700, 900))
		else:
			show = show


		b, g, r = cv2.split(show)
		ret, g = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)
		ret, b = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)

		Xb, Yb = np.where((b > 0) | (g > 0))

		r[Xb, Yb] = 0
		# r = cv2.medianBlur(r, 7)
		ret, r = cv2.threshold(r, 60, 255, cv2.THRESH_BINARY)
		cv2.imshow("r", r)
		cv2.imshow("show", show)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	test_bags()

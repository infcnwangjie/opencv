import cv2
import numpy

from app.core.target_detect.pointlocation import PointLocationService


def comput_test():
	cap = cv2.VideoCapture('C:/work/imgs/test02.mp4')
	positionservice = PointLocationService()
	while 1:
		ret, frame = cap.read()
		positionservice.img = frame
		location_info = positionservice.computelocations()
		print(location_info)
		cv2.imshow("cap", positionservice.img)
		if cv2.waitKey(100) & 0xff == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


def sdk_read():
	pass

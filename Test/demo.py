import cv2
import numpy

from app.core.target_detect.pointlocation import PointLocationService
from app.core.video.sdk import  SdkHandle


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


def skdavi_test():
	cap = SdkHandle()
	while True:
		frame = cap.read()
		cv2.resize(frame, frame)
		cv2.imshow("cap", frame)
		if cv2.waitKey(100) & 0xff == ord('q'):
			break
	del cap


def sdk_test():
	image = SdkHandle().read()
	cv2.namedWindow("result", 0)
	cv2.imshow("result", image)
	cv2.waitKey(0)


if __name__ == '__main__':
	skdavi_test()

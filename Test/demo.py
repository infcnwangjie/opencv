import cv2
import numpy

from app.core.location.locationservice import PointLocationService
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
#
# def read_num():
# 	import pytesseract
# 	from PIL import Image
# 	#
# 	# print(pytesseract.tesseract_version())  # print tesseract-ocr version
# 	# print(pytesseract.get_languages())  # prints tessdata path and list of available languages
#
# 	image = Image.open(r'C:\work\imgs\test\10.png')
# 	print(pytesseract.image_to_string(image))  # print ocr text from image
# 	# or
# 	# print(pytesseract.file_to_text('sample.jpg'))

if __name__ == '__main__':
	comput_test()

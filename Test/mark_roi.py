# import the necessary packages
import argparse
import cv2


# drawing = False
# mode = True  # 如果mode为true绘制矩形。按下'm' 变成绘制曲线。

def mouse_click(event, x, y, flags, param):
	# grab references to the global variables

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		cv2.putText(img=image,text="({},{})".format(x,y),org=(x+50,y+50),color=(0, 255, 127),thickness=3)
		# cv2.circle(img=image, center=(x, y), color=(0, 255, 127), radius=3, thickness=2)
	elif event == cv2.EVENT_MOUSEMOVE and event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
	elif event == cv2.EVENT_LBUTTONUP:
		pass


def set_roi(image):
	clone = image.copy()
	cv2.namedWindow("image",0)
	cv2.setMouseCallback("image", mouse_click)
	# keep looping until the 'q' key is pressed
	cv2.imshow("image", image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	image = cv2.imread("2020-04-10-15-26-22test.bmp")
	set_roi(image)

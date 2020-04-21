# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image = None

drawing = False
mode = True  # 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
ix, iy = -1, -1

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, image, drawing
	copyimage = image.copy()

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		refPt = [(x, y)]
		cv2.circle(img=image, center=(x, y), color=(0, 255, 127), radius=3, thickness=2)
	elif event == cv2.EVENT_MOUSEMOVE and event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		drawing == False
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.circle(img=image, center=(refPt[0][0], refPt[1][1]), color=(0, 255, 127), radius=3, thickness=2)
		cv2.circle(img=image, center=(refPt[1][0], refPt[1][1]), color=(0, 255, 127), radius=3, thickness=2)
		cv2.circle(img=image, center=(refPt[1][0], refPt[0][1]), color=(0, 255, 127), radius=3, thickness=2)
		cv2.imshow("image", image)


def set_roi(image):
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	# keep looping until the 'q' key is pressed
	cv2.imshow("image", image)
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) > 1:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[-1][0]:refPt[-1][0]]
		cv2.imshow("ROI", roi)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	image = cv2.imread("2020-04-10-15-26-22test.bmp")
	set_roi(image)

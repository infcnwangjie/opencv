import numpy as np
import cv2

# cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
# ret, frame = cap.read()
# setup initial location of window
r, h, c, w = 67, 100, 540, 100  # simply hardcoded the values
track_window = (c, r, w, h)
# set up the ROI for tracking
# img=cv2.imread("imgs/test/without_person.bmp")

roi = cv2.imread("../imgs/test/6.png")
# roi = img[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array(((61, 83, 31))), np.array((81, 255, 250)))
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

img_detect = cv2.imread("../imgs/test/bag5.bmp")
hsv = cv2.cvtColor(img_detect, cv2.COLOR_BGR2HSV)
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

# apply meanshift to get the new location
ret, track_window = cv2.meanShift(dst, track_window, term_crit)

# Draw it on image
x, y, w, h = track_window
img2 = cv2.rectangle(img_detect, (x, y), (x + w, y + h), 255, 2)
cv2.namedWindow('img2', 0)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

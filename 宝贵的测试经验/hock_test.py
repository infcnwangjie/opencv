import cv2
import numpy as np


def green_range(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# cv2.imshow("hsv",hsv)
	# green_low, green_high = [35, 43, 46], [77, 255, 255]
	green_low, green_high = [35, 43, 46], [77, 255, 255]
	green_min, green_max = np.array(green_low), np.array(green_high)
	green_mask = cv2.inRange(hsv, green_min, green_max)
	green_mask = cv2.medianBlur(green_mask, 3)
	green_ret, foreground = cv2.threshold(green_mask, 60, 255, cv2.THRESH_BINARY)
	# disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# foreground = cv2.filter2D(foreground, -1, disc)
	green_contours, _hierarchy = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return green_contours, foreground


img = cv2.imread("D:/Image_20201020104352597.bmp")
img = cv2.resize(img, (400, 600))
r = None
g = None
b = None
# thresh_copy[Xb, Yb] = 0
b, g, r = cv2.split(img)


def getposgray(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print("g", g[y, x])


# foreground = cv2.medianBlur(red_binary, 3)
cv2.imshow("bag", img)
# ret, r1 = cv2.threshold(r, 100, 255, cv2.THRESH_BINARY)
# ret, b1 = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)
#
Xb, Yb = np.where((b > 200) | (r > 200))
#
g[Xb, Yb] = 0

# ret, g = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
# g = cv2.medianBlur(r, 3)
result = np.concatenate((r, g, b), axis=1)
cv2.imshow("rgb", result)

# ret, g = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY)
# cv2.imshow("g",g)

# ret, not_green = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("not_green",not_green)

# result = cv2.bitwise_and(r, r, mask=not_green)
# result = cv2.medianBlur(r, 3)
# cv2.imshow("b",b)
cv2.imshow("g", g)

cv2.setMouseCallback("g", getposgray)

ign_, g_f = green_range(img)
cv2.imshow("g_f", g_f)

cv2.waitKey(0)
cv2.destroyAllWindows()

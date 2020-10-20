import cv2
import numpy as np


img = cv2.imread("D:/bag3.png")
r=None
g=None
b=None

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)
ret, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
Xb, Yb = np.where(gray > 0)
# thresh_copy[Xb, Yb] = 0
b, g, r = cv2.split(img)


def getposgray(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("r", r[y, x])


# foreground = cv2.medianBlur(red_binary, 3)
cv2.imshow("bag", img)
ret, g = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)
ret, b = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)

Xb, Yb = np.where((b > 0)|(g > 0))

r[Xb, Yb] = 0



ret, r = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
r = cv2.medianBlur(r, 3)
cv2.imshow("b",b)
cv2.imshow("r",r)
cv2.imshow("g",g)
# ret, g = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY)
# cv2.imshow("g",g)

# ret, not_green = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("not_green",not_green)

# result = cv2.bitwise_and(r, r, mask=not_green)
# result = cv2.medianBlur(r, 3)
# cv2.imshow("b",b)
# cv2.imshow("result",result)
cv2.setMouseCallback("r",getposgray)


cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

from core.target_detect.digitdetect import DigitDetector

img = cv2.imread('C:/work/imgs/test/juxing.png', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, 3, 2)
# img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# # 0 1 6               8 9
cnt_a, cnt_b, cnt_c = contours[0], contours[1], contours[2]
print(cv2.matchShapes(cnt_c,cnt_b,1,0.0)) #0和8：1.5
# print(cv2.matchShapes(cnt_b, cnt_b, 1, 0.0))  # 0.0
# print(cv2.matchShapes(cnt_b, cnt_c, 1, 0.0))  # 1 和6  ：0.44
# print(cv2.matchShapes(cnt_b, cnt_a, 1, 0.0)) #1和0 ：1.39

#
# tool = DigitDetector()
# tool.practise()
# for contour in  contours:
# 	[digit_point_x, digit_point_y, digit_contor_width, digit_contor_height] = cv2.boundingRect(
# 		contour)
# 	roi = thresh[digit_point_y:digit_point_y + digit_contor_height,
# 	      digit_point_x:digit_point_x + digit_contor_width]
# 	result=tool.readnum(roi)
# 	print(result)


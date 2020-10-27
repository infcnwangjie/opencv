import numpy as np
import cv2

img = cv2.imread('E:/xianweiqi.png',0)
cv2.imshow('image',img)
cv2.resize(img,(300,300))
k = cv2.waitKey(0)
## k = cv2.waitKey(0) & 0xFF  # 64位机器
if k == 27:         # 按下esc时，退出
    cv2.destroyAllWindows()
elif k == ord('s'): # 按下s键时保存并退出
    cv2.imwrite('ianzha.png',img)
    cv2.destroyAllWindows()
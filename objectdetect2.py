import numpy as np
import cv2 as cv

roi = cv.imread('imgs/test/7.jpg')
gray = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)

target = cv.imread('imgs/test/bag3.bmp')
grayt = cv.cvtColor(target,cv.COLOR_BGR2GRAY)

# calculating object histogram，目标直方图
# roihist = cv.calcHist([gray],[0, 1], None, [180, 256], [0, 180, 0, 256] )
roihist=cv.calcHist([gray],[0],None,[256],[0,256])
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX) #归一化处理
dst = cv.calcBackProject([grayt],[0,1],roihist,[0,180,0,256],1) #反向投影

# Now convolute with circular disc
#圆盘算子卷积
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# threshold and binary，二值化。参见《图像阈值》
ret,thresh = cv.threshold(dst,50,255,0)
# thresh = cv.merge((thresh,thresh,thresh)) #三通道图像，因此这里使用merge变成3 通道
# res = cv.bitwise_and(target,thresh) #位操作，获取图像中的一部分，参见《核心操作》
# res = np.vstack((target,thresh,res))

cv.namedWindow("result",0)
cv.imshow("result",thresh)
cv.waitKey(0)
cv.destroyAllWindows()
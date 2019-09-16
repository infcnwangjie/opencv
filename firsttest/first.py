#coding=utf-8
import cv2 as cv
#img = cv.imread("/root/test.jpg")

#cv.namedWindow("Image")
#cv.imshow("Image",img)
#cv.waitKey(0)
#cv2.destroyAllWindows() 
src=cv.imread('Hydrangeas.jpg')
cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)  
cv.imshow('input image',src)  



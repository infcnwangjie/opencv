#coding=utf-8
import cv2 as cv
src=cv.imread('Hydrangeas.jpg')
# print(src[0,0])
# print(src.item(0,0,0))
# src.itemset((0,0,0),50)
# print(src.item(0,0,0))

# srcroi=src[100:200,100:200]
# src[200:300,200:300]=srcroi
cv.namedWindow('input image',cv.WINDOW_AUTOSIZE)
cv.imshow('input image',src)
cv.waitKey(0)



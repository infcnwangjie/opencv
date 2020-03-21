#opencv模板匹配----单目标匹配
import cv2
#读取目标图片
target = cv2.imread("../imgs/test/bag5.bmp")
#读取模板图片
template = cv2.imread("../imgs/test/8.png")
#获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
#执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target_gray,template_gray,cv2.TM_SQDIFF_NORMED)
#归一化处理
cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
#寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#匹配值转换为字符串
#对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
#对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
#绘制矩形边框，将匹配区域标注出来
#min_loc：矩形定点
#(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
#(0,0,225)：矩形的边框颜色；2：矩形边框宽度
# cv2.rectangle(target,min_loc,(max_loc[0]+twidth,max_loc[1]+theight),(0,0,225),2)
#显示结果,并将匹配值显示在标题栏上


print(min_loc)
print(max_loc)
cv2.circle(target,min_loc,5,color=(255,0,0),thickness=3)
cv2.putText(target,"min_point", (min_loc[0] + 50, min_loc[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 225), 2)

cv2.circle(target,max_loc,5,thickness=3,color=(0,255,0))
cv2.putText(target,"max_point", (max_loc[0] + 50, max_loc[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

cv2.namedWindow("match_result",0)
cv2.imshow("match_result",target)
cv2.waitKey()
cv2.destroyAllWindows()
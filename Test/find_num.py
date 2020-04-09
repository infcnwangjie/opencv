import cv2

import numpy as np

def enhance_histrg( img):
	Imin, Imax = cv2.minMaxLoc(img)[:2]
	# 使用numpy计算
	# Imax = np.max(img)
	# Imin = np.min(img)
	Omin, Omax = 0, 255
	# 计算a和b的值
	a = float(Omax - Omin) / (Imax - Imin)
	b = Omin - a * Imin
	out = a * img + b
	out = out.astype(np.uint8)
	return out

def orb_match():
	img1 = cv2.imread("C:/work/imgs/test/1.jpg")  # 导入灰度图像
	img2 = cv2.imread("C:/work/imgs/test/final.jpg")

	detector = cv2.ORB_create()

	kp1 = detector.detect(img1, None)
	kp2 = detector.detect(img2, None)
	kp1, des1 = detector.compute(img1, kp1)
	kp2, des2 = detector.compute(img2, kp2)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	for matchitem in matches:
		print(matchitem)
	# cv2.imshow("t",des2)
	# matches = sorted(matches, key=lambda x: x.distance)  # 据距离来排序
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=0)
	cv2.namedWindow("orbTest",0)
	cv2.imshow('orbTest', img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def tempate_detect():
	import cv2
	# 读取模板图片
	template = cv2.imread("C:/work/imgs/test/1.jpg")
	# 读取目标图片
	target = cv2.imread("C:/work/imgs/test/final.jpg")
	# 获得模板图片的高宽尺寸
	theight, twidth = template.shape[:2]

	template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
	template_gray=enhance_histrg(template_gray)
	target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
	target_gray=enhance_histrg(target_gray)
	cv2.imshow("target_gray",target_gray)
	# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
	result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_SQDIFF_NORMED)
	# 归一化处理
	cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
	# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

	# 匹配值转换为字符串
	# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
	# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
	strmin_val = str(min_val)
	# 绘制矩形边框，将匹配区域标注出来
	# min_loc：矩形定点
	# (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
	# (0,0,225)：矩形的边框颜色；2：矩形边框宽度
	# cv2.rectangle(target,min_loc,(max_loc[0]+twidth,max_loc[1]+theight),(0,0,225),2)
	# 显示结果,并将匹配值显示在标题栏上

	print(min_loc)
	print(max_loc)
	cv2.circle(target, min_loc, 5, color=(255, 0, 0), thickness=3)
	cv2.putText(target, "min_point", (min_loc[0] + 50, min_loc[1] + 10),
	            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 225), 2)

	cv2.circle(target, max_loc, 5, thickness=3, color=(0, 255, 0))
	cv2.putText(target, "max_point", (max_loc[0] + 50, max_loc[1] + 10),
	            cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 105, 225), 2)

	cv2.namedWindow("match_result", 0)
	cv2.imshow("match_result", target)
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
    # orb_match()
    tempate_detect()
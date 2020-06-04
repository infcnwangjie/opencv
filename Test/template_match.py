#opencv模板匹配----单目标匹配
import cv2


def test1():
    # 读取目标图片
    target = cv2.imread("D:/PIC/Image_20200602111317712.bmp")
    # 读取模板图片
    template = cv2.imread("D:/PIC/hock.png")
    # 获得模板图片的高宽尺寸
    theight, twidth = template.shape[:2]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_SQDIFF_NORMED)
    # 归一化处理
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
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


def template_demo():
    tpl = cv2.imread("D:/PIC/hock.png")
    # target = cv2.imread("D:/PIC/Image_20200602111317712.bmp")
    # target=cv2.imread("D:/PIC/Image_20200602111240982.bmp")
    target = cv2.imread("D:/PIC/com.bmp")

    target=cv2.resize(target,(700,900))
    cv2.imshow("template image", tpl)
    cv2.imshow("target image", target)
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED] # 各种匹配算法
    th, tw = tpl.shape[:2]# 获取模板图像的高宽
    for md in methods:
        result = cv2.matchTemplate(target, tpl, md)
        # result是我们各种算法下匹配后的图像
        # cv.imshow("%s"%md,result)
        # 获取的是每种公式中计算出来的值，每个像素点都对应一个值
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if md == cv2.TM_SQDIFF_NORMED:
            tl = min_loc  # tl是左上角点
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)  # 右下点
        cv2.rectangle(target, tl, br, (0, 0, 255), 2)# 画矩形
        cv2.imshow("match-%s" % md, target)


def hist_compare():
    target = cv2.imread('D:/PIC/target.png')
    target = cv2.resize(target, (100, 100))
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    targethist = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(targethist, targethist, 0, 255, cv2.NORM_MINMAX)

    # roi图片，就想要找的的图片
    roi = cv2.imread('D:/PIC/hock.png')
    roi = cv2.resize(roi, (100, 100))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 计算目标直方图
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 归一化，参数为原图像和输出图像，归一化后值全部在2到255范围
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)

    result=cv2.compareHist(targethist,roihist, cv2.HISTCMP_CORREL)
    print(result)




def hist_demo():
    # 目标搜索图片
    target = cv2.imread('D:/PIC/com.bmp')
    target = cv2.resize(target, (700, 900))
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # roi图片，就想要找的的图片
    roi = cv2.imread('D:/PIC/hock.png')
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 计算目标直方图
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 归一化，参数为原图像和输出图像，归一化后值全部在2到255范围
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

    # 卷积连接分散的点
    # disc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dst = cv2.filter2D(dst, -1, disc)

    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    # 使用merge变成通道图像
    # thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.medianBlur(thresh, 3)

    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100: continue
        rect = cv2.boundingRect(contour)
        rect_x, rect_y, rect_w, rect_h = rect
        cv2.rectangle(target, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color=(0, 255, 255),
                      thickness=1)

    # 蒙板
    # res = cv2.bitwise_and(target, thresh)
    # 矩阵按列合并,就是把target,thresh和res三个图片横着拼在一起
    # cv2.imwrite('res.jpg', res)
    # 显示图像
    cv2.imshow('1', thresh)
    cv2.imshow('target', target)
    cv2.waitKey(0)


def main():
    # test1()
    # src = cv2.imread("./1.png")  # 读取图片
    # cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)  # 创建GUI窗口,形式为自适应
    # cv2.imshow("input image", src)  # 通过名字将图像和窗口联系
    # hist_demo()
    # template_demo()
    hist_compare()
    cv2.waitKey(0)  # 等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
    cv2.destroyAllWindows()  # 销毁所有窗口




if __name__ == '__main__':
    main()

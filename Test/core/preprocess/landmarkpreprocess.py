import cv2
import numpy as np


# 发现
class LandMarkPreprocess(object):
    def __init__(self, img):
        if isinstance(img, str):
            self.img = cv2.imread(img)
        else:
            self.img = img

    # 直方图正规化
    def enhance_histrg(self, img):
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

    def enhance_gmchange(self, img):
        # 图像归一化
        fi = img / 255.0
        # 伽马变换
        gamma = 0.4
        # power(x1, x2):对x1中的每个元素求x2次方。不会改变x1上午shape。
        out = np.power(fi, gamma)
        return out

    def sharper(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        dst = cv2.filter2D(image, -1, kernel=kernel)
        return dst

    # 普通二值化操作
    def find_contours_byeasyway(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        # gray=self.enhance_histrg(gray)
        ret, binary = cv2.threshold(gray, 70, 150, cv2.THRESH_BINARY)  # 灰度阈值
        # ret, binary = cv2.threshold(gray, 80, 150, cv2.THRESH_BINARY)  # 灰度阈值

        # binary=self.enhance_gmchange(binary)

        # binary = self.sharper(binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda c: 2500 < cv2.contourArea(c) < 20000, contours))
        # contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

        cv2.namedWindow("all_contours_binary", 0)
        cv2.imshow("all_contours_binary", binary)
        return contours, binary

    # 找到地标的轮廓
    def find_contours_bylandmark_colorrange(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        colorlow = (61, 83, 31)
        colorhigh = (81, 255, 250)
        colormin, colormax = np.array(colorlow), np.array(colorhigh)
        # 去除颜色范围外的其余颜色
        mask = cv2.inRange(hsv, colormin, colormax)
        # mask = cv2.erode(mask, None, iterations=3)


        ret, binary = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)
        # binary = binary.copy()
        # rows = binary.shape[0]
        # cols = binary.shape[1]
        # for i in range(rows):
        # 	for j in range(cols):
        # 		binary[i][j] = 3 * np.math.log(1 + binary[i][j])
        binary = self.sharper(binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     roi=binary[y+1:y+h-1,x+1:x+w-1]
        #     ret, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV)
        #     binary[y+1:y+h-1,x+1:x+w-1]=roi_binary
        # binary = self.sharper(binary)
        # cv2.namedWindow("landmark_mask", 0)
        # cv2.imshow("landmark_mask", binary)
        return contours, binary

    # 找到袋子轮廓
    def find_contours_bybagcolorrange(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 70, 150, cv2.THRESH_BINARY)  # 灰度阈值
        # 对binary去噪，腐蚀与膨胀
        binary = cv2.erode(binary, None, iterations=3)
        cv2.namedWindow("bag_detect", 0)
        cv2.imshow("bag_detect", binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
        return contours[0:20], binary

    # 找到蓝色小车轮廓
    def find_contours_bybluecarcolorrange(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        colormin, colormax = np.array([100, 43, 46]), np.array([124, 255, 255])
        # 去除颜色范围外的其余颜色
        mask = cv2.inRange(hsv, colormin, colormax)
        # kernel=np.uint8(np.zeros((5,5)))
        # for x in range(5):
        # 	kernel[0,2]=1
        # 	kernel[2,x]=1
        # cv2.dilate(mask,kernel)
        # mask = cv2.erode(mask, None, iterations=2)
        # cv2.namedWindow("mask", 0)
        # cv2.imshow("mask", mask)
        ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        temp_binary = np.zeros_like(binary)
        # binary.where()
        cv2.namedWindow("car_detect", 0)
        cv2.imshow("car_detect", binary)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
        return contours[0:10], binary

    # 移除图像干扰，从地标检测道路上清除任何与地标无关的障碍物
    def remove_noises(self):
        # 第一步去除袋子的干扰
        img1 = self.img.copy()
        landmark_contours, landmark_binary = self.find_contours_bylandmark_colorrange()
        # result1 = cv2.drawContours(img1, landmark_contours, -1,
        #                            (0, 255, 0), 3)
        # bag_contours, bag_binary = self.find_contours_bybagcolorrange()
        # result2 = cv2.drawContours(result1, bag_contours, -1,
        #                            (0, 0, 255), 3)
        # bluecar_contours, bluecar_binary = self.find_contours_bybluecarcolorrange()
        # result3 = cv2.drawContours(result2, bluecar_contours, -1,
        #                            (255, 0, 0), 3)

        easy_contours, easy_binary = self.find_contours_byeasyway()
        result = cv2.drawContours(img1, landmark_contours, -1,
                                  (0, 0, 255), 3)
        # finaly = np.zeros_like(landmark_binary)
        # 发现hsv检测出来的方式，轮廓大致保存完整，不完整的部分，让普通二进制的方式补充即可
        # 1：该方法找到地标的位置，从灰度图像上拷贝合适的像素过来，相对来说可以
        # for landmarkcontour in landmark_contours:
        #     if cv2.contourArea(landmarkcontour) > 900:
        #         x, y, w, h = cv2.boundingRect(landmarkcontour)
        #         finaly[y + 1:y + h, x + 1:x + w] = easy_binary[y + 1:y + h, x + 1:x + w]
        #         cv2.rectangle(finaly, (x - 8, y - 8), (x + w + 8, y + h + 8), color=255, thickness=3)
        # 2：除了地标位置的像素保留不动，其余的像素都是地标二进制的

        cv2.namedWindow("landmarkrange", 0)
        cv2.imshow("landmarkrange", result)

        return landmark_binary

    # 获取已处理过的二值化图像
    @property
    def processedimg(self):
        img = self.remove_noises()
        return img


if __name__ == '__main__':
    process = LandMarkPreprocess(img="bag5.bmp")
    # img = process.processedimg
    img = process.processedimg
    cv2.namedWindow("finaly", 0)
    cv2.imshow("finaly", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

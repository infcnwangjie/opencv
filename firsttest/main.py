import cv2
import numpy as np
from scipy import ndimage


def gaussblur():
    kernel_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]
                           ])
    img = cv2.imread("img.png", 0)
    k3 = ndimage.convolve(img, kernel_3x3)
    cv2.imshow("3*3", k3)
    cv2.imshow("img3", img)
    k5 = ndimage.convolve(img, kernel_3x3)
    cv2.imshow("img5", img)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    g_hpf = img - blurred
    cv2.imshow("g-hpf", g_hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()


def edge_detect():
    src = cv2.imread("img.png")
    blurredsrc = cv2.medianBlur(src, ksize=5)
    graysrc = cv2.cvtColor(blurredsrc, cv2.COLOR_RGB2GRAY)
    dest = graysrc
    cv2.Laplacian(graysrc, cv2.CV_8U, dest, ksize=5)
    cv2.imshow("result", dest)
    cv2.waitKey()
    cv2.destroyAllWindows()


def canny_test():
    src = cv2.imread("img.png")
    cv2.imshow("result", cv2.Canny(src, 200, 300))
    cv2.waitKey()
    cv2.destroyAllWindows()


def find_contours():
    img = cv2.imread("img.png")
    cv2.imshow("img", img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('thresh', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("thresh", binary)

    diffimg = gray - binary

    cv2.namedWindow('diffimg', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("diffimg", diffimg)

    contours, hierarchy = cv2.findContours(diffimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    cv2.namedWindow('contours', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("contours", image)
    # # cv2.imshow("contours",img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # gaussblur()
    # edge_detect()
    # canny_test()
    find_contours()

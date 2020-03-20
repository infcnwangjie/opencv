import cv2


def orb_match():
    template1 = cv2.imread("imgs/land_mark/6.png")  # 导入灰度图像

    obj_img = cv2.imread("imgs/test/bag3.bmp")

    detector = cv2.ORB_create()

    kp_obj = detector.detect(obj_img, None)
    kp_obj, des_obj = detector.compute(obj_img, kp_obj)

    kp1_t = detector.detect(template1, None)
    kp1_t, des1_t = detector.compute(template1, kp1_t)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1_t, des_obj)
    img3 = cv2.drawMatches(template1, kp1_t, obj_img, kp_obj, matches[:10], None, flags=0)

   
   
    # for matchitem in matches:
    #     print(matchitem)
    
    cv2.namedWindow("orbTest", 0)
    cv2.imshow('orbTest', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


orb_match()

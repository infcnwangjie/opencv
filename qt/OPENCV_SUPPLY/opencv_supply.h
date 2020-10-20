#ifndef OPENCV_SUPPLY_H
#define OPENCV_SUPPLY_H

#include "OPENCV_SUPPLY_global.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include <opencv2\imgproc\types_c.h>

class OPENCV_SUPPLY_EXPORT OPENCV_SUPPLY
{
public:
    OPENCV_SUPPLY();
};


extern "C" OPENCV_SUPPLY_EXPORT void helloWorld();
extern "C" OPENCV_SUPPLY_EXPORT int add(int a,int b);

//extern "C" OPENCV_SUPPLY_EXPORT typedef struct ImageBase {
//    int w;                   //图像的宽
//    int h;                    //图像的高
//    int c;                    //通道数
//    unsigned char * data;    //我们要写python和c++交互的数据结构，
//}ImageMeta;

extern "C"   OPENCV_SUPPLY_EXPORT   unsigned char * find_it( unsigned char * image_meta ,
                                                             unsigned char * model_data,
                                                             int src_w ,int src_h,
                                                             int model_w,int model_h);

extern "C"   OPENCV_SUPPLY_EXPORT   double same_rate(unsigned char * img1_data,unsigned char * img2_data, int img1_w,int img1_h,int img2_w,int img2_h);


extern "C"   OPENCV_SUPPLY_EXPORT uchar* cpp_canny(int height, int width, uchar* data) ;

extern "C"   OPENCV_SUPPLY_EXPORT int category_code(unsigned char * test_img_data,int img_h,int img_w);


#endif // OPENCV_SUPPLY_H

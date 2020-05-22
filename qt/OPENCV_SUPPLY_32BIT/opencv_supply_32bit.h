#ifndef OPENCV_SUPPLY_32BIT_H
#define OPENCV_SUPPLY_32BIT_H
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include <opencv2\imgproc\types_c.h>
#include "OPENCV_SUPPLY_32BIT_global.h"

class OPENCV_SUPPLY_32BIT_EXPORT OPENCV_SUPPLY_32BIT
{
public:
    OPENCV_SUPPLY_32BIT();
};

extern "C" OPENCV_SUPPLY_32BIT_EXPORT void helloWorld();
extern "C" OPENCV_SUPPLY_32BIT_EXPORT int add(int a,int b);

//extern "C" OPENCV_SUPPLY_EXPORT typedef struct ImageBase {
//    int w;                   //图像的宽
//    int h;                    //图像的高
//    int c;                    //通道数
//    unsigned char * data;    //我们要写python和c++交互的数据结构，
//}ImageMeta;

extern "C"   OPENCV_SUPPLY_32BIT_EXPORT   unsigned char * find_it( unsigned char * image_meta ,
                                                             unsigned char * model_data,
                                                             int src_w ,int src_h,
                                                             int model_w,int model_h);

extern "C"   OPENCV_SUPPLY_32BIT_EXPORT uchar* cpp_canny(int height, int width, uchar* data) ;

extern "C"   OPENCV_SUPPLY_32BIT_EXPORT void release(uchar* data) {
    free(data);
}
#endif // OPENCV_SUPPLY_32BIT_H

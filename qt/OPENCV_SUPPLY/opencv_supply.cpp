#include "opencv_supply.h"
#include <iostream>
using namespace std;
using namespace cv;

OPENCV_SUPPLY::OPENCV_SUPPLY()
{
}

void helloWorld(){
    cout << "hello world!"<<endl;
}

int add(int a , int b){
    return a+b;
}
unsigned char * find_it( unsigned char * image_meta , unsigned char * model_data,
                                                            int src_w ,int src_h,
                         int model_w,int model_h)
{
//    Mat image = Mat::zeros(Size(src_w,src_h),CV_8UC3);
//    image.data=image_meta;

    cv::Mat image(src_h, src_w, CV_8UC3, image_meta);

//    Mat model = Mat::zeros(Size(model_w,model_h),CV_8UC3);
//    model.data=model_data;

    cv::Mat model(model_h, model_w, CV_8UC3, model_data);

    Mat model_hsv, image_hsv;
    cvtColor(model, model_hsv, COLOR_BGR2HSV);
    cvtColor(image, image_hsv, COLOR_BGR2HSV);

    // 定义直方图参数与属性
    int h_bins = 32; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    Mat roiHist;

    calcHist(&model_hsv, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
    normalize(roiHist, roiHist, 0, 255, NORM_MINMAX, -1, Mat());

    //计算反向投影
    Mat backproj;
    calcBackProject(&image_hsv, 1, channels, roiHist, backproj, ranges, 1.0);


   return  backproj.data;
}

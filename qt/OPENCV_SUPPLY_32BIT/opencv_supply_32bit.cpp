#include "opencv_supply_32bit.h"
using namespace std;
using namespace cv;
OPENCV_SUPPLY_32BIT::OPENCV_SUPPLY_32BIT()
{
}

void helloWorld(){
    cout << "hello world!"<<endl;
}

int add(int a , int b){
    return a+b;
}
unsigned char  *  find_it( unsigned char * image_meta , unsigned char * model_data,
                                                            int src_w ,int src_h,
                         int model_w,int model_h)
{

    cv::Mat image(src_h, src_w, CV_8UC3, image_meta);



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
    normalize(roiHist, roiHist, 0, 256, NORM_MINMAX, -1, Mat());

    //计算反向投影
    Mat backproj;
    calcBackProject(&image_hsv, 1, channels, roiHist, backproj, ranges, 1.0);

   uchar* buffer = (uchar*)malloc(sizeof(uchar)*src_h*src_w);
   memcpy(buffer, backproj.data, src_h*src_w);
   return buffer;
}


uchar* cpp_canny(int height, int width, uchar* data) {
   cv::Mat src(height, width, CV_8UC1, data);
   cv::Mat dst;
   Canny(src, dst, 100, 200);

   uchar* buffer = (uchar*)malloc(sizeof(uchar)*height*width);
   memcpy(buffer, dst.data, height*width);
   return buffer;

}

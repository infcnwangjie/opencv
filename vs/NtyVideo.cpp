// NtyVideo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

static const int width = 50, height =50;

typedef struct NearLandMark {

	int col;
	int row;
	Mat slide_img;
	double similarity;
}NearLandMark;

typedef struct LandMarkRoi {

	Mat img;
	string name;
	double rate=0;
	int x;
	int y;
	bool find;
	NearLandMark landmark;

} LandMarkRoi;


inline Mat calcBackProject(Mat &srcimg,Mat & roi ) {
	int h_bins = 32; int s_bins = 32;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat roiHist;

	calcHist(&roi, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
	normalize(roiHist, roiHist, 0, 255, NORM_MINMAX, -1, Mat());

	//计算反向投影
	MatND backproj;
	calcBackProject(&srcimg, 1, channels, roiHist, backproj, ranges, 1.0);
	return backproj;
}





int main()
{
	//0表示将图像转换为灰度再返回
	Mat srcimg = imread("d:/2020-05-12-10-53-30test.bmp");

	imshow("原图1", srcimg);
	Mat src_resize;
	cv::resize(srcimg, src_resize, cv::Size(700, 900));


	Mat srcHsvImg; 
	cv::cvtColor(src_resize, srcHsvImg,cv::COLOR_BGR2HSV);



	LandMarkRoi GREEN_R;
	GREEN_R.img = imread("D:/T_G_R_.png");
	cv::cvtColor(GREEN_R.img, GREEN_R.img, cv::COLOR_BGR2HSV);
	cv::resize(GREEN_R.img, GREEN_R.img, cv::Size(50, 50));
	GREEN_R.find = false;
	GREEN_R.name = "GREEN_R";



	vector<LandMarkRoi> rois;
	rois.push_back(GREEN_R);


	float hue_range[] = { 0, 256 };
	const float* ranges = { hue_range };
	for (LandMarkRoi itemroi : rois) {
		
		Mat backproj = calcBackProject(srcHsvImg,itemroi.img);
		threshold(backproj, backproj, 50, 255, 0);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		cv::findContours(backproj,contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		//drawContours(src_resize, contours, -1, Scalar(255, 255, 255));
		for (auto contour : contours) {
			auto area = cv::contourArea(contour);

			if (area < 100)continue;
			auto rect=cv::boundingRect(contour);

			int x=rect.x, y=rect.y, width=rect.width, height=rect.height;

			rectangle(src_resize, Rect(x, y , width, height), (0, 255, 255), 1);
		}

		//cv::imshow("backproj", backproj);

		//rectangle(src_resize, Rect(right.x-50, right.y-50, 60,  60), (0, 255, 255), 1);
	}

	//flip(src_resize, red_green_2.img);
	cv::imshow("find_it", src_resize);
	waitKey(0);
	//waitKey(0);
	return 0;
}


#include "ocr.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>





using namespace cv;
using namespace std;

void OcrTest::find_words(){
    cout<<"ok"<<endl;
    try{
        Mat  google = imread("D:/wj/google.PNG");
        imshow("google.PNG",google);

        cv::Mat gray;
        cv::cvtColor(google, gray, cv::COLOR_BGR2GRAY);
        // ...other image pre-processing here...

        imshow("gray.PNG",gray);


        waitKey(4000);
    }
    catch(Exception e){
        cout<<e.what()<<endl;
    }
}

#ifndef ROI_H
#define ROI_H

#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <QObject>

#include <QMessageBox>
using namespace std;
using namespace cv;
class RoiTemplate{
public:
    RoiTemplate(){}
    RoiTemplate(Mat srcimg,string name):img(srcimg),name(name){}
    ~RoiTemplate(){}

    void setName(string name);
    string getName();
    void setRate(double rate);
    double getRate();
    void setPosition(int x=0,int y=0);
    int * getPosition();
    void setFind(bool find);
    bool getFind();
    Mat getImg(){
        return this->img;
    }

private:
    Mat img;
    string name;
    double rate;
    int x;
    int y;
    bool find=false;
};

class ImgProcessCore :public QObject{
    Q_OBJECT
public:
    ImgProcessCore(){}

    void flip(Mat& img, vector<RoiTemplate> & roi_objs) {
        int rows = img.rows;
        int cols = img.cols;

        Mat roi;

        vector<RoiTemplate>::iterator roi_template_iterator;

        for (int i = 0; i < 650; i++)
        {
            for (int j = 90; j < 200; j++)
            {
                for (roi_template_iterator = roi_objs.begin(); roi_template_iterator < roi_objs.end(); ++roi_template_iterator) {

                    roi = img(Rect(i, j, 50, 50));
                    double result = similar(roi, roi_template_iterator->getImg());


                    if (result > 0.5 && result>roi_template_iterator->getRate() && roi_template_iterator->getFind()==false) {

                        cv::rectangle(img, Rect(i, j, 70, 70), Scalar(0, 0, 255));
                        if (result > 0.5) {
                            roi_template_iterator->setFind( true);
                            break;

                        }
                        roi_template_iterator->setRate( result);
                        roi_template_iterator->setPosition(j,i);
                    }
                    else {
                        continue;
                    }



                }


            }
        }

    }


signals: void finish(const QString & msg);

private:
    inline double similar(Mat img1, Mat img2) {
        //定义变量
        Mat dstHist1,dsHist2;
        int dims = 1;
        float hranges[] = { 0, 256 };
        const float* ranges[] = { hranges };   // 这里需要为const类型
        int size = 256;
        int channels = 0;

        //计算图像的直方图
        calcHist(&img1, 1, &channels, Mat(), dstHist1, dims, &size, ranges);
        calcHist(&img2, 1, &channels, Mat(), dsHist2, dims, &size, ranges);
        double result = cv::compareHist(dstHist1, dsHist2, 0);
        return result;

    }


};



#endif // ROI_H

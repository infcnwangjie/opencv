#include "opencv_supply.h"
#include<string>
#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>

#include <vector>
#include<fstream>
#include<sstream>
#include <QDebug>
using namespace cv;
using namespace ml;
using namespace std;
OPENCV_SUPPLY::OPENCV_SUPPLY()
{
}


class HogUtil{
public:
    HogUtil( int width=64,int height=64):img_width(width),img_height(height){
        this->dectector= HOGDescriptor(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    }

    vector<float> compute_img(Mat dstImage){
        vector<float> descriptors;
        vector<Point> locations;
        dectector.compute(dstImage, descriptors,  Size(8, 8));
        return descriptors;
    }

private:
    HOGDescriptor dectector;
    int img_width;
    int img_height;

};

class SvmUtil{

public:
    SvmUtil(){
        this->svm=SVM::create();
        this->svm->setType(SVM::C_SVC);
        this->svm->setKernel(SVM::POLY);
        this->svm->setDegree(1.0);
        this->svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,100,1e-6));
        this->hogUtil=HogUtil();
    }


    void train(Mat trainDataMat,Mat labelsMat){
        this->svm->train(trainDataMat,ROW_SAMPLE,labelsMat);
    }

    void train(string posdir,string posimgsaddresstxt,string ngdir,string ngimgsaddresstxt){

        ifstream finPos(posdir + posimgsaddresstxt); //正样本地址txt文件
        ifstream finNeg(ngdir + ngimgsaddresstxt); //负样本地址txt文件
        //        qDebug("read_neg_pos_data");

        stringstream stringformat;


        //---------------------------逐个读取正样本图片，生成HOG描述子-------------
        string pos_img_name,neg_img_name;
        vector<vector<float>> pos_descriptors;
        vector<vector<float>> neg_descriptors;




        vector<int> label_values;
        while(getline(finPos, pos_img_name)){
            pos_img_name = posdir + pos_img_name;
            Mat src = imread(pos_img_name);
            Mat result_img;
            resize(src,result_img,Size(64,64));

            auto descript= hogUtil.compute_img(result_img);
            pos_descriptors.push_back(descript);
            label_values.push_back(1);
        }

        while(getline(finNeg, neg_img_name)){
            neg_img_name = ngdir + neg_img_name;
            Mat src = imread(neg_img_name);
            Mat result_img;
            resize(src,result_img,Size(64,64));

            auto descript= hogUtil.compute_img(result_img);
            neg_descriptors.push_back(descript);
            label_values.push_back(0);
        }

        //获取向量维度
        auto get_dim=[pos_descriptors,neg_descriptors](string flag)->int{

            int dim=0;
            //            cout<<flag<<endl;
            if(pos_descriptors.size()>0){
                dim= pos_descriptors[0].size();
                return dim;
            }
            return dim;
        };




        int total_train_num=pos_descriptors.size()+neg_descriptors.size();
        int featureDim=get_dim("pos");

        stringformat<<"read_neg_pos_data"<<featureDim;

        //        qDebug(stringformat.str().data());

        Mat sampleFeatureMat=Mat::zeros(total_train_num, featureDim, CV_32FC1); // 所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数为HOG描述子维数
        Mat sampleLabelMat=Mat::zeros(total_train_num, 1, CV_32SC1);   // 所有训练样本的类别向量，行数等于所有样本的个数， 列数为1： 1表示有目标，-1表示无目标


        for(int index=0;index<total_train_num;index++){
            if(index<pos_descriptors.size()){
                sampleLabelMat.at<int>(index, 0) = 1;  //正样本标签矩阵 值为1

                for (int i = 0; i<featureDim; i++)
                    sampleFeatureMat.at<float>(index, i) = pos_descriptors[index][i];

            }else{
                sampleLabelMat.at<int>(index, 0) = 0;  //负样本标签矩阵 值为0
                int neg_index=index-pos_descriptors.size();
                //                cout<<neg_index<<endl;
                for (int i = 0; i<featureDim; i++)
                    sampleFeatureMat.at<float>(index, i) = neg_descriptors[neg_index][i];
            }



        }

        this->svm->train(sampleFeatureMat,ROW_SAMPLE,sampleLabelMat);

    }


    float predict(Mat sampleMat){


        Mat src_resize;
        resize(sampleMat,src_resize,Size(64,64));
        //        imshow("predict",src_resize);

        auto descriptor=hogUtil.compute_img(src_resize);
        auto get_dim=[descriptor]{

            int dim=0;
            dim= descriptor.size();
            return dim;
        };
        int dim=  get_dim();

        Mat testFeatureMat=Mat::zeros(1, get_dim(), CV_32FC1);

        for (int i = 0; i<dim; i++)
            testFeatureMat.at<float>(0, i) =descriptor[i];

        float response = this->svm->predict(testFeatureMat);
        return response;


    }

    float predict(string test_img_path){
        Mat src = imread(test_img_path);
        Mat src_resize;
        resize(src,src_resize,Size(64,64));
        //        imshow("predict",src_resize);

        auto descriptor=hogUtil.compute_img(src_resize);
        auto get_dim=[descriptor]{

            int dim=0;
            dim= descriptor.size();
            return dim;
        };
        int dim=  get_dim();

        Mat testFeatureMat=Mat::zeros(1, get_dim(), CV_32FC1);

        for (int i = 0; i<dim; i++)
            testFeatureMat.at<float>(0, i) =descriptor[i];

        float response = this->svm->predict(testFeatureMat);
        return response;
    }

private:
    Ptr<SVM> svm;
    HogUtil hogUtil;
    vector<vector<float>> pos_descriptors;
    vector<vector<float>> neg_descriptors;

};



double same_rate(unsigned char * img1_data,unsigned char * img2_data, int img1_w,int img1_h,int img2_w,int img2_h){
    double rate=0;

    cv::Mat img1(img1_h, img1_w, CV_8UC3, img1_data);

    cv::Mat img2(img2_h, img2_w, CV_8UC3, img2_data);

    Mat img1_hsv, img2_hsv;
    cv::cvtColor(img1, img1_hsv, COLOR_BGR2HSV);
    cv::cvtColor(img2, img2_hsv, COLOR_BGR2HSV);

    // 定义直方图参数与属性
    int h_bins = 32; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    cv:: Mat img1Hist,img2Hist;

    cv::calcHist(&img1_hsv, 1, channels, Mat(), img1Hist, 2, histSize, ranges, true, false);
    cv::normalize(img1Hist, img1Hist, 0, 256, NORM_MINMAX, -1, Mat());

    cv::calcHist(&img2_hsv, 1, channels, Mat(), img2Hist, 2, histSize, ranges, true, false);
    cv::normalize(img2Hist, img2Hist, 0, 256, NORM_MINMAX, -1, Mat());
    rate= cv::compareHist(img1Hist,img2Hist,HISTCMP_CORREL);

    return rate;
}


unsigned char  *  find_it( unsigned char * image_meta , unsigned char * model_data,
                           int src_w ,int src_h,
                           int model_w,int model_h)
{

    cv::Mat image(src_h, src_w, CV_8UC3, image_meta);



    cv::Mat model(model_h, model_w, CV_8UC3, model_data);

    cv::Mat model_hsv, image_hsv;
    cv::cvtColor(model, model_hsv, COLOR_BGR2HSV);
    cv::cvtColor(image, image_hsv, COLOR_BGR2HSV);

    // 定义直方图参数与属性
    int h_bins = 32; int s_bins = 32;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    cv::Mat roiHist;

    calcHist(&model_hsv, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
    normalize(roiHist, roiHist, 0, 256, NORM_MINMAX, -1, Mat());

    //计算反向投影
    cv:: Mat backproj;
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


int category_code(unsigned char * test_img_data,int img_h,int img_w){

    auto svm= SvmUtil();

    svm.train("D:/NTY_IMG_PROCESS/TRAIN/POS/","imgs.txt","D:/NTY_IMG_PROCESS/TRAIN/NEG/","imgs.txt");
     cv::Mat image(img_h, img_w, CV_8UC3, test_img_data);
    int result=svm.predict(image);
    return result;
}

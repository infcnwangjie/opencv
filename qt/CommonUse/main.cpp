#include "mainwindow.h"

#include <QApplication>


#include "ocr.h"
#include<string>
#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
using namespace cv;
using namespace ml;

#include <vector>
#include<fstream>
#include<sstream>
using namespace std;
#include <QDebug>

#include<regex>


void test_regex(){
    std::string pattern{ "\\d{3}-\\d{8}|\\d{4}-\\d{7}" }; // fixed telephone
    std::regex re(pattern);

    bool ret = std::regex_match("010-12345678", re);
    cout<<"010-12345678 match regex \\d{3}-\\d{8}|\\d{4}-\\d{7} :"<<ret<<endl;


    std::regex pattern1("\\d{4}");
    std::string content("hello_2018");
    std::smatch result;
    if (std::regex_match(content, result, pattern1)) {
        std::cout << result[0]<<endl;
    }


    std::regex pattern2("\\d{4}");
    std::string content2("hello_2018 by_2017");
    std::smatch result2;
    //匹配任意一个子串，匹配到就返回，result2[1]是空，说明仅有一个对象
    if (std::regex_search(content2, result2, pattern2)) {
        std::cout << result2[0]<<endl;
        std::cout << result2[1]<<endl;
    }

    std::regex pattern3("\\d{4}");
    std::string content3("hello_2018 by_2017");

    std::string result3 = std::regex_replace(content3, pattern3, "everyone");
    std::cout << result3<<endl;

    //当匹配多个，并且返回多个，只能用sregex_token_iterator
    string str = "boq@yahoo.com, boqian@gmail.com; bo@hotmail.com";

    //regex e("[[:punct:]]+");  // 空格，数字，字母以外的可打印字符
    //regex e("[ [:punct:]]+");
    regex e("([[:w:]]+)@([[:w:]]+)\.com");

    sregex_token_iterator pos(str.cbegin(), str.cend(), e, 0);    //最后一个参数指定打印匹配结果的哪一部分，0表达整个匹配字符串，1表示第1个子串，-1表示没有匹配的部分
    sregex_token_iterator end;  // 默认构造定义了past-the-end迭代器
    for (; pos!=end; pos++) {
        cout << "Matched:  " << *pos << endl;
    }
    cout << "=============================\n\n";


    //当匹配多个，并且返回多个，只能用sregex_token_iterator
    std::string str1("hello_2018 by_2017");

    regex e1("\\d{4}");

    sregex_iterator pos1(str1.cbegin(), str1.cend(), e1);    //最后一个参数指定打印匹配结果的哪一部分，0表达整个匹配字符串，1表示第1个子串，-1表示没有匹配的部分
    sregex_iterator end1;  // 默认构造定义了past-the-end迭代器
    for (; pos1!=end1; pos1++) {
        cout << "Matched:  " << pos1->str(0) << endl;
    }
    cout << "=============================\n\n";


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
    template<typename t>
    void print_a(t a){
        cout<<a<<endl;
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
        float response = this->svm->predict(sampleMat);
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


void test_hog(){

    Mat srcImage = imread("D:/wj/px_train/pos/0.bmp");
    Mat grayImage, dstImage;
    cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
    resize(grayImage, dstImage, Size(64, 128));

    HOGDescriptor dectector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    // HOG的描述子
    vector<float> descriptors;
    vector<Point> locations;
    dectector.compute(dstImage, descriptors, Size(0, 0), Size(0, 0), locations);
    printf("HOG descriptors: %d\n", descriptors.size());
    imshow("dstImage", dstImage);
    waitKey();


}

void test_svm(){
    //使用SVM进行分类

    //训练数据
    const int sampleSum = 12;
    int labels[sampleSum] = {0,0,1,1,1,1,1,2,2,3,3,3};
    float trainData[sampleSum][2] = {{79,50},{74,175},{173,416},{133,216},{222,333},{192,283},{118,400},{278,156},{394,117},{340,296},{351,437},{479,218}};
    Mat trainDataMat(sampleSum,2,CV_32FC1,trainData);
    Mat labelsMat(sampleSum,1,CV_32SC1,labels);
    //建立模型
    Ptr<SVM> model = SVM::create();
    model->setType(SVM::C_SVC);
    model->setKernel(SVM::POLY);
    model->setDegree(1.0);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,100,1e-6));
    model->train(trainDataMat,ROW_SAMPLE,labelsMat);

    //对每个像素点进行分类
    Mat showImg = Mat::zeros(512, 512, CV_8UC3);
    for (int i=0; i<showImg.rows; i++)
    {
        for (int j=0; j<showImg.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
            float response = model->predict(sampleMat);
            for (int label = 0; label < sampleSum; label++)
            {
                if (response == labels[label])
                {
                    RNG rng1(labels[label]);
                    showImg.at<Vec3b>(i, j) = Vec3b(rng1.uniform(0,255),rng1.uniform(0,255),rng1.uniform(0,255));
                }
                RNG rng2(3-labels[label]);
                circle(showImg,Point(trainData[label][0],trainData[label][1]),8,Scalar(rng2.uniform(0,255),rng2.uniform(0,255),rng2.uniform(0,255)),-1,-1);
            }
        }
    }
    //绘制出支持向量
    Mat supportVectors = model->getSupportVectors();
    for (int i = 0; i < supportVectors.rows; ++i)
    {
        const float* sv = supportVectors.ptr<float>(i);
        circle(showImg,Point(sv[0],sv[1]),8,Scalar(0,255,0),2,8);
    }
    //测试
    Mat testLabels;
    float testData[3][2] = {{456,123},{258,147},{58,111}};
    Mat testDataMat(3,2,CV_32FC1,testData);
    model->predict(testDataMat, testLabels);
    std::cout <<"testLabels：\n"<<testLabels<<std::endl;
    imshow("output", showImg);
    waitKey();

}




//hog——begin




//hog_end






void final_test_svm_hog(){

    auto svm= SvmUtil();
    svm.train("D:/wj/px_train/upright/","addresses.txt","D:/wj/px_train/handstand/","addresses.txt");
    int result=svm.predict("D:/wj/px_test/20200622_121041_2_0.bmp");
    cout<<result<<endl;
}


void test_memcpy(){

    vector<int> a{1,2,3,4,5};
    vector<int>b;
    memcpy((vector<int> *)&b,(vector<int> *)&a,sizeof(a));
    for(int item : b){
        cout<<item<<endl;
    }

    int c(10);
    cout<<c<<endl;
    int d{11};
    cout<<d<<endl;

}


int main(int argc, char *argv[])
{
    auto hog= HogUtil(56,23);
    hog.print_a(56);
    //   final_test_svm_hog();
    //    test_memcpy();
//    test_regex();
    //    train_SVM_HOG();
    //    test_hog();
    //        test_svm();
    //    OcrTest testhandle;
    //    testhandle.find_words();
    //    QApplication a(argc, argv);
    //    MainWindow w;
    //    w.show();
    //    return a.exec();
    return 0;
}

#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include <opencv2\imgproc\types_c.h>

#include <QMessageBox>
#include <vector>
using namespace  std;



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->pushButton, SIGNAL(QPushButton::clicked), this, SLOT(this->on_pushButton_clicked));

    QObject::connect(&this->processcore,&ImgProcessCore::finish,&this->afterProcessSlot,&AfterProcessSlot::receiveMsg);

}

MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::on_pushButton_clicked()
{
    using namespace cv;

    Mat image = imread("C:/work/imgs/2020-04-10-15-26-22test.bmp");//括号里为自己图像的路径
    Mat src_resize;
    cv::resize(image, src_resize, cv::Size(700, 900));


    Mat  red_green_roi_img = imread("C:/work/imgs/red_green_2.png");
    cv::resize(red_green_roi_img, red_green_roi_img, cv::Size(60, 60));
    RoiTemplate red_green_2(red_green_roi_img,"red_green_right");



    Mat  red_green_2r_roi_img = imread("C:/work/imgs/red_green_2r.png");
    cv::resize(red_green_2r_roi_img, red_green_2r_roi_img, cv::Size(60, 60));
    RoiTemplate red_green_2r(red_green_2r_roi_img,"red_green_right");


    Mat  red_yellow3_img = imread("C:/work/imgs/red_yellow3.png");
    cv::resize(red_yellow3_img, red_yellow3_img, cv::Size(60, 60));
    RoiTemplate red_yellow3(red_yellow3_img,"red_yellow3");
    vector<RoiTemplate> landmark_rois{red_green_2,red_green_2r,red_yellow3};


    this->processcore.flip(src_resize,landmark_rois);
    cvtColor(src_resize,image,CV_BGR2RGB);
    this->ui->label->resize(image.rows,image.cols);
     QImage img = QImage((const unsigned char*)(image.data),image.cols,image.rows, image.cols*image.channels(),  QImage::Format_RGB888);
    img.scaled(this->ui->label->size(),Qt::KeepAspectRatio );
    this->ui->label->clear();
    this->ui->label->setPixmap(QPixmap::fromImage(img));


    emit this->processcore.finish("cat show finish"); //发射信号

//    waitKey(0);
}

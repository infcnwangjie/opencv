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
using namespace cv;


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


    //    Mat image = imread("C:/work/imgs/2020-04-10-15-26-22test.bmp");//括号里为自己图像的路径
    //    Mat src_resize;
    //    cv::resize(image, src_resize, cv::Size(700, 900));

    VideoCapture cap1(0);

    cap1.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
    //    cap2.set(CAP_PROP_FRAME_WIDTH, 1280);
    //    cap2.set(CAP_PROP_FRAME_HEIGHT, 480);


    Mat frame;


    Mat left,right;
    while (true)
    {
        cap1 >> frame;

        if(frame.data!=NULL){
            left=frame(cv::Range(0,480),cv::Range(0,640));
            right=frame(cv::Range(0,480 ),cv::Range(640,1280));
            imshow("left", left);
            imshow("right", left);
            //            this->ui->label->resize(480,1280);
            //            QImage img = QImage((const unsigned char*)(frame1.data),frame1.cols,frame1.rows, frame1.cols*frame1.channels(),  QImage::Format_RGB888);
            //            img.scaled(this->ui->label->size(),Qt::KeepAspectRatio );
            //            this->ui->label->clear();
            //            this->ui->label->setPixmap(QPixmap::fromImage(img));
        }





        waitKey(60);
    }
    left.release();
    right.release();

    cap1.release();



    emit this->processcore.finish("cat show finish"); //发射信号


}

#include "main_widget.h"
#include "ui_main_widget.h"
#include <cstdio>
using namespace cv;
using std::cout;
using std::endl;


MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent),ui(new Ui::MainWidget),m_cycle(-1)
{
    ui->setupUi(this);
//    loadVideo("test1.avi");
        loadVideo("test1.mp4");
}

MainWidget::~MainWidget()
{
    delete ui;
}
void MainWidget::loadVideo(QString s)
{
    m_video.open(s.toStdString());
    if(!m_video.isOpened())
    {
        std::cout<<"Error in video input !!"<<std::endl;
    }
    m_algorithm.setVideo(&m_video);
    m_algorithm.initProcess();
}
void MainWidget::loadVideo(int s)
{
    m_video.open(s);
    if(!m_video.isOpened())
    {
        std::cout<<"Error in video input !!"<<std::endl;
    }
    m_algorithm.setVideo(&m_video);
    m_algorithm.initProcess();
}
void MainWidget::keyPressEvent(QKeyEvent *key)
{
    switch(key->key())
    {
    case Qt::Key_B:
        processVideo(m_video);
        break;
    default:
        break;
    }
}
void MainWidget::processVideo(VideoCapture& video)
{
    m_algorithm.processNextFrame();
//    showImage(m_klt_detector.getCurFrame(),ui->m_main_label);
    showImage(m_algorithm.getDeltaFrame(),ui->m_main_label);
    showImage(m_algorithm.getPreFrame(),ui->m_main_label_2);
    this->update();
}


void MainWidget::processImage()
{
    //    //    m_cycle++;
    //    Mat image;
    //    Vec<uchar,3> temp;
    //    uchar average;
    ////    int a=m_image.rows;
    ////    int b=m_image.cols;
    //    //    m_image.copyTo(image);
    //    switch (m_cycle)
    //    {
    //    case 1:
    //        GaussianBlur(m_show_image,image,Size(7,7),2,2);
    //        showImage(image);
    //        break;
    //    case 2:
    //        image.create(m_show_image.rows,m_show_image.cols,CV_8UC3);
    //        for(int i=0;i<image.rows;i++)
    //        {
    //            for(int j=0;j<image.cols;j++)
    //            {
    //                temp = m_show_image.at< Vec<uchar,3> >(i,j);
    //                average = (temp[0]+temp[1]+temp[2])/3;
    //                image.at< Vec<uchar,3> >(i,j)= Vec<uchar,3>(average,average,average);
    //            }
    //        }
    //        showImage(image);
    //        break;
    //    case 3:
    //        image.create(m_show_image.rows,m_image.cols,CV_8UC3);
    //        for(int i=0;i<image.rows;i++)
    //        {
    //            for(int j=0;j<image.cols;j++)
    //            {
    //                temp = m_image.at< Vec<uchar,3> >(i,j);
    //                average = (temp[0]+temp[1]+temp[2])/3;
    //                image.at< Vec<uchar,3> >(i,j)= Vec<uchar,3>(average,average,average);
    //            }
    //        }
    //        GaussianBlur(image,image,Size(7,7),2,2);
    //        showImage(image);
    //        break;
    //    default:
    //        break;
    //    }

}

void MainWidget::showImage(Mat image)
{
    // convert image to 3 channel
    switch(image.channels())
    {
    case 1:
        cvtColor(image, image, CV_GRAY2BGR);
        break;
    default:
        break;
    }

    QImage img = QImage(image.data,image.cols,image.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();
    img = img.scaled(ui->m_main_label->width(),ui->m_main_label->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp;
    temp.convertFromImage(img);
    ui->m_main_label->setPixmap(temp);
}
void MainWidget::showImage(Mat image, QLabel *label)
{
    // convert image to 3 channel
    Mat show;
    switch(image.channels())
    {
    case 1:
        cvtColor(image, show, CV_GRAY2BGR);
        break;
    default:
        show = image;
        break;
    }

    QImage img = QImage(show.data,show.cols,show.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();
    img = img.scaled(ui->m_main_label->width(),ui->m_main_label->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp;
    temp.convertFromImage(img);
    label->setPixmap(temp);
}

#include "main_widget.h"
#include "ui_main_widget.h"
#include <cstdio>
using namespace cv;
using std::cout;
using std::endl;
namespace
{
RNG rng(12345);
static const float resize_scale = 0.5;
static const int features_num = 500;
}

MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent),ui(new Ui::MainWidget),m_cycle(-1)
{
    ui->setupUi(this);
    loadImage("lena.jpg");
    loadVideo("test.mp4");
    //    connect(ui->m_button_start,SIGNAL(clicked()),this,SLOT(processImage()));
}

MainWidget::~MainWidget()
{
    delete ui;
}
void MainWidget::loadImage(QString s)
{
    //    m_image = imread(s.toStdString());
    //    showImage(m_image);
}
void MainWidget::loadVideo(QString s)
{
//    m_video.open(1);
    m_video.open(s.toStdString());
    if(!m_video.isOpened())
    {
        std::cout<<"Error in video input !!"<<std::endl;
    }
    processVideo(m_video);
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
    m_cycle++;
    if(!m_cycle)
    {
        if( video.read(m_prev_frame) && video.read(m_current_frame))
        {

            cout<<"Video Process Start !!!"<<endl;
            cv::resize(m_prev_frame,m_prev_frame,Size(),resize_scale,resize_scale);
            cv::resize(m_current_frame,m_current_frame,Size(),resize_scale,resize_scale);
            cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);


        }
        else
        {
            std::cout<<"Error in video show !!!"<<std::endl;
        }
    }
    else
    {
        // get video pic
        m_prev_frame = m_current_frame.clone();
        if( !video.read(m_current_frame)||!video.read(m_current_frame))
        {

            std::cout<<"Error in video show !!!"<<std::endl;
            return;
        }
        cv::resize(m_current_frame,m_current_frame,Size(),resize_scale,resize_scale);


        // to gray scale
        swap(m_prev_frame_gray,m_current_frame_gray);
        cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);

        // get features points
        vector<Point2f> points1 = get_KLT_features(m_prev_frame_gray,features_num);

        // get motion line
        vector<Point2f> points2 = points1;
        vector<unsigned char> features_status;
        vector<float> features_error;
        calcOpticalFlowPyrLK(m_prev_frame, m_current_frame,
                             points1, points2, features_status, features_error,
                             Size(21,21), 5,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                             OPTFLOW_LK_GET_MIN_EIGENVALS,1e-4);
        for( size_t i=0;i<points1.size();i++)
        {
            if( !features_status[i] || features_error[i]>550 )
            {
                continue;
            }
            line(m_prev_frame,points1[i],points2[i],Scalar(255,255,255),1,8,0);
        }
        showImage(m_current_frame,ui->m_main_label);
        showImage(m_prev_frame,ui->m_main_label_2);
    }
}
vector<Point2f> MainWidget::get_KLT_features(Mat image,int maxCorners)
{
    Mat temp;
    if(image.channels()==3)
    {
        cvtColor(image,temp,CV_BGR2GRAY);
    }
    else
    {
        temp=image.clone();
    }


    /// Parameters for Shi-Tomasi algorithm
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    /// Apply corner detection
    goodFeaturesToTrack( temp,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         useHarrisDetector,
                         k );


    /// find corner sub-pixel
    cornerSubPix(temp, corners, Size(blockSize,blockSize), Size(-1,-1),
                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,20,0.03));


    /// Draw corners detected
    std::cout<<"** Number of corners detected: "<<corners.size()<<std::endl;
//    int r = 4;
//    for( int i = 0; i < corners.size(); i++ )
//    { circle( image, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255),
//                                           rng.uniform(0,255)), -1, 8, 0 ); }


    return corners;
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
    this->update();
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
    this->update();
}

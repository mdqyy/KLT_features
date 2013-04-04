#include "my_algorithm.h"


using namespace cv;
using std::cout;
using std::endl;
namespace
{
RNG rng(12345);
static const cv::Size frame_size = cv::Size(428,240);
//static const cv::Size frame_size = cv::Size(0,0);
static const float resize_scale = 1.0/4;
static const int features_num = 500;
}

KLTDetector::KLTDetector(cv::VideoCapture* video)
{
    m_video_ptr = video;
}
void KLTDetector::setVideo(VideoCapture *video)
{
    m_video_ptr = video;
}

void KLTDetector::initProcess()
{
    if(m_video_ptr==NULL)
    {
        cout<<"No Video Input !!!"<<endl;
    }
    if( m_video_ptr->read(m_prev_frame) && m_video_ptr->read(m_current_frame))
    {

        cout<<"Video Process Start !!!"<<endl;
        cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR);
//        cv::resize(m_current_frame,m_current_frame,Size(1280,720));
        cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);
    }
    else
    {
        std::cout<<"Error in video show !!!"<<std::endl;
    }
}

void KLTDetector::processNextFrame()
{
    if(m_video_ptr==NULL)
    {
        cout<<"No Video Input !!!"<<endl;
    }
    // get next frame
    m_prev_frame = m_current_frame.clone();
    if( !m_video_ptr->read(m_current_frame))//||!m_video_ptr->read(m_current_frame))
    {
        std::cout<<"Error in video show !!!"<<std::endl;
        return;
    }
    cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR);
//    cv::resize(m_current_frame,m_current_frame,Size(1280,720));

    // to gray scale
    swap(m_prev_frame_gray,m_current_frame_gray);
    cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);

    // get features points
    m_pre_klt_points = get_KLT_features(m_prev_frame_gray,features_num);

    // get motion line
    std::vector<unsigned char> features_status;
    std::vector<float> features_error;
    calcOpticalFlowPyrLK(m_prev_frame, m_current_frame,
                         m_pre_klt_points, m_cur_klt_points, features_status, features_error,
                         Size(5,5), 5,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         OPTFLOW_LK_GET_MIN_EIGENVALS,1e-4);

    for( size_t i=0;i<m_pre_klt_points.size();i++)
    {
        if( !features_status[i] || features_error[i]>550 )
        {
            continue;
        }
        line(m_prev_frame,m_pre_klt_points[i],m_cur_klt_points[i],Scalar(255,255,255),1,8,0);
    }
}
std::vector<Point2f> KLTDetector::get_KLT_features(Mat image,int maxCorners)
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


    // Parameters for Shi-Tomasi algorithm
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    // Apply corner detection
    goodFeaturesToTrack( temp,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         useHarrisDetector,
                         k );


    // find corner sub-pixel
    cornerSubPix(temp, corners, Size(blockSize,blockSize), Size(-1,-1),
                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,20,0.03));


    // output corners number
    std::cout<<"** Number of corners detected: "<<corners.size()<<std::endl;
    return corners;
}
std::vector<cv::Point2f> KLTDetector::getPreKLTPoints()
{
    return m_pre_klt_points;
}
std::vector<cv::Point2f> KLTDetector::getCurKLTPoints()
{
    return m_cur_klt_points;
}
cv::Mat KLTDetector::getPreFrame()
{
    return m_prev_frame;
}
cv::Mat KLTDetector::getCurFrame()
{
    return m_current_frame;
}

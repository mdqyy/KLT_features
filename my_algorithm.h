#ifndef ALGORITHM_H
#define ALGORITHM_H
#include "opencv2/opencv.hpp"
#include <vector>

class KLTDetector
{
public:
    KLTDetector(cv::VideoCapture* video=NULL);
    void setVideo(cv::VideoCapture* video);
    void initProcess();
    void processNextFrame();


    std::vector<cv::Point2f> getPreKLTPoints();
    std::vector<cv::Point2f> getCurKLTPoints();
    cv::Mat getPreFrame();
    cv::Mat getCurFrame();
private:
    cv::VideoCapture* m_video_ptr;
    cv::Mat m_current_frame;
    cv::Mat m_current_frame_gray;
    cv::Mat m_prev_frame;
    cv::Mat m_prev_frame_gray;

    std::vector<cv::Point2f> m_pre_klt_points;
    std::vector<cv::Point2f> m_cur_klt_points;
private:

    std::vector<cv::Point2f> get_KLT_features(cv::Mat image,int maxCorners);


};

#endif // ALGORITHM_H

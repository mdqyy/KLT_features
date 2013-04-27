#ifndef ALGORITHM_H
#define ALGORITHM_H
#include "particle_filter.h"
#include "opencv2/opencv.hpp"
#include <vector>





class LinearTransformManager
{
public:
    LinearTransformManager();
    void setTransformInfo(cv::Mat input_vec,cv::Mat output_vec);
    void solveTransform();
    cv::Mat getTransformMat();
    inline cv::Mat getOutputFromTransform(cv::Mat input)
    {
      cv::Mat retv = m_transform_mat*input;
      return retv;
    }
private:


    cv::Mat m_input_vec; // n examples * m features
    cv::Mat m_output_vec; // k outputs * n examples
    cv::Mat m_transform_mat; //  k outputs * m features
};

class KLTDetector
{
public:
    KLTDetector();
    void setCurrentFrame(cv::Mat frame);
    void doTrackingKLT();

    std::vector<cv::Point2f> getPreKLTPoints();
    std::vector<cv::Point2f> getCurKLTPoints();
    cv::Mat getPreFrame();
    cv::Mat getCurFrame();
private:
    cv::Mat m_current_frame;
    cv::Mat m_current_frame_gray;
    cv::Mat m_prev_frame;
    cv::Mat m_prev_frame_gray;

    std::vector<cv::Point2f> m_pre_klt_points;
    std::vector<cv::Point2f> m_cur_klt_points;

private:

    std::vector<cv::Point2f> get_KLT_features(cv::Mat image,int maxCorners);

};

class MyAlgorithm
{
public :
    MyAlgorithm();
    ~MyAlgorithm();

    cv::Mat getPreFrame();
    cv::Mat getCurFrame();
    cv::Mat getDeltaFrame();

    void setVideo(cv::VideoCapture* video);
    void initProcess();
    void processNextFrame(); // temp function


private:
    void doKLT();
    void doTransform();
    void doPF();
private:
    cv::VideoCapture* m_video_ptr;
    cv::Mat m_prev_frame;
    cv::Mat m_current_frame;
    cv::Mat m_delta_frame;

    KLTDetector m_klt_detector;
    LinearTransformManager m_lt_manager;
    ParticleFilter m_particle_filter;
};



#endif // ALGORITHM_H

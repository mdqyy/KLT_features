#ifndef ALGORITHM_H
#define ALGORITHM_H
#include "particle_filter.h"
#include "kmeans/KMlocal.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <vector>





class LinearTransformManager
{
public:
    LinearTransformManager();
    void setTransformInfo(cv::Mat input_vec_x,cv::Mat input_vec_y,cv::Mat output_vec_x,cv::Mat output_vec_y);
    void solveTransform();
    cv::Mat getTransformMat();
    inline cv::Mat getOutputFromTransform(cv::Mat input,bool is_x_flag = true)
    {
        if(is_x_flag)
        {
            cv::Mat retv = m_transform_mat_x*input;
            return retv;
        }
        else
        {
            cv::Mat retv = m_transform_mat_y*input;
            return retv;
        }
    }
private:


    cv::Mat m_input_vec_x; // n examples * 1 feature x
    cv::Mat m_input_vec_y;// n examples * 1 feature y
    cv::Mat m_output_vec_x; //1 x outputs * n examples
    cv::Mat m_output_vec_y; //1 y outputs * n examples
    cv::Mat m_transform_mat_x; //  1 outputs * m features
    cv::Mat m_transform_mat_y; //  1 outputs * m features
};

class KLTDetector
{
public:
    KLTDetector();
    void setCurrentFrame(cv::Mat frame);
    void doTrackingKLT();

    const std::vector<cv::Point2f>& getPreKLTPoints();
    const std::vector<cv::Point2f>& getCurKLTPoints();
    cv::Mat getPreFrame();
    cv::Mat getCurFrame();
private:
    // cpu format
    cv::Mat m_current_frame;
    cv::Mat m_current_frame_gray;
    cv::Mat m_prev_frame;
    cv::Mat m_prev_frame_gray;

    // gpu format
    cv::gpu::GpuMat m_current_frame_gpu;
    cv::gpu::GpuMat m_prev_frame_gpu;
    cv::gpu::GpuMat m_current_frame_gray_gpu;
    cv::gpu::GpuMat m_prev_frame_gray_gpu;
    //    inline bool isFlowCorrect(cv::Point2f u)
    //    {
    //        return !cv::cvIsNaN(u.x) && !cv::cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
    //    }

    // klt points
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
    void doTransform(int fnum);
    void doPF();
    void doClustering();
    void sumDeltaFrame();
private:
    cv::VideoCapture* m_video_ptr;
    cv::Mat m_raw_frame;
    cv::Mat m_prev_frame;
    cv::Mat m_current_frame;
    cv::Mat m_delta_frame;

    // get KLT features and track
    KLTDetector m_klt_detector;
    vector<cv::Point2f> m_cur_klt_points;
    vector<cv::Point2f> m_pre_klt_points;

    // get linear tranformation
    LinearTransformManager m_lt_manager;

    // do particle filter
    ParticleFilter m_particle_filter;

    // clustering
    KMdata m_km_dataPts;		// allocate data storage
    vector<cv::Point2f> m_mid_p;
    cv::Point2f m_min_mid_p;
    vector<double> m_std_error;
    double m_min_std_error;

};



#endif // ALGORITHM_H

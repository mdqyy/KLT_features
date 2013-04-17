#include "my_algorithm.h"


using namespace cv;
using std::cout;
using std::endl;
namespace
{
RNG rng(12345);
static const bool c_down_twice = false;
static const int gauss_kernel_size = 43;
static const float gauss_sigma = 100;

static const cv::Size frame_size = cv::Size(320,180);
//static const cv::Size frame_size = cv::Size(0,0);
static const float resize_scale = 1.0/2;
static const int feature_points_num = 500;
static const int features_num = 4;
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
        GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),gauss_sigma);
        cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
        if(c_down_twice)
        {
            GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),10);
            cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
        }
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
    GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),gauss_sigma);
    cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
    if(c_down_twice)
    {
        GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),10);
        cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
    }

    // to gray scale
    swap(m_prev_frame_gray,m_current_frame_gray);
    cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);

    // get features points
    m_pre_klt_points = get_KLT_features(m_prev_frame_gray,feature_points_num);

    // get motion line
    std::vector<unsigned char> features_status;
    std::vector<float> features_error;
    calcOpticalFlowPyrLK(m_prev_frame, m_current_frame,
                         m_pre_klt_points, m_cur_klt_points, features_status, features_error,
                         Size(5,5), 5,
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         OPTFLOW_LK_GET_MIN_EIGENVALS,1e-4);
    std::vector<Point2f> prev_points,cur_points;
    for( size_t i=0;i<m_pre_klt_points.size();i++)
    {
        if( !features_status[i] || features_error[i]>50 )
        {
            continue;
        }
        prev_points.push_back(m_pre_klt_points.at(i));
        cur_points.push_back(m_cur_klt_points.at(i));
    }
    m_pre_klt_points = prev_points;
    m_cur_klt_points = cur_points;


    //get transform matrix
    Mat input(m_cur_klt_points.size(),features_num,CV_64F);
    Mat output(2,m_pre_klt_points.size(),CV_64F);
    for(size_t t=0;t<m_pre_klt_points.size();t++)
    {

        input.at<double>(t,0) = m_cur_klt_points.at(t).x;
        input.at<double>(t,1) = m_cur_klt_points.at(t).y;
        input.at<double>(t,2) = 1;
        if(features_num == 4)
        {
            input.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;
        }
        output.at<double>(0,t) = m_pre_klt_points.at(t).x;
        output.at<double>(1,t) = m_pre_klt_points.at(t).y;
    }
    ltmanager.setTransformInfo(input,output);
    ltmanager.solveTransform();

    // get delta frame
    m_delta_frame = Mat(m_current_frame.rows,m_current_frame.cols,CV_8UC1);
    Mat trans = ltmanager.getTransformMat();
    Mat output_p;
    int a[2];
    Mat inp(features_num,1,CV_64F);
    for(size_t i=0;i<m_delta_frame.rows;i++)
    {
        for(size_t j=0;j<m_delta_frame.cols;j++)
        {
            inp.at<double>(0,0)=j;//x,cols
            inp.at<double>(1,0)=i;//y,rows
            inp.at<double>(2,0)=1;
            inp.at<double>(3,0)=j*i;
            //            output_p = trans*inp;
            output_p = ltmanager.getOutputFromTransform(inp);
            a[0]=output_p.at<double>(0,0);//x,cols
            a[0]=min(a[0],m_delta_frame.cols-1);
            a[0]=max(a[0],0);
            a[1]=output_p.at<double>(1,0);//y,rows
            a[1]=min(a[1],m_delta_frame.rows-1);
            a[1]=max(a[1],1);

            if(a[0]==0||a[0]==m_delta_frame.cols-1
                    ||a[1]==0||a[1]==m_delta_frame.rows-1)
            {
                m_delta_frame.at<uchar>(i,j)=0 ;
                //                m_delta_frame.at<Vec3b>(i,j)=Vec3b(0,0,0) ;
            }
            else
            {
                Vec3b p1= m_current_frame.at<Vec3b>(i,j);
                Vec3b p2= m_prev_frame.at<Vec3b>(a[1],a[0]);
                Vec3b dd;
                int hehe=0;
                for(int k=0;k<3;k++)
                {
                    dd[k]=(p1[k]>p2[k])?(p1[k]-p2[k]):(p2[k]-p1[k]);
                    hehe+=dd[k];
                }
                //                m_delta_frame.at<Vec3b>(i,j)=dd;
                m_delta_frame.at<uchar>(i,j)=hehe/3;
            }
        }
    }

    for( size_t i=0;i<m_pre_klt_points.size();i++)
    {
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
cv::Mat KLTDetector::getDeltaFrame()
{
    return m_delta_frame;
}






LinearTransformManager::LinearTransformManager()
{

}
void LinearTransformManager::setTransformInfo(cv::Mat input_vec,cv::Mat output_vec)
{
    m_input_vec = input_vec;
    m_output_vec = output_vec;
}
void LinearTransformManager::solveTransform()
{
    Mat result;
    result = (m_input_vec.t()*m_input_vec).inv(DECOMP_CHOLESKY)*m_input_vec.t()*m_output_vec.t(); //m * k
    m_transform_mat = result.t();// k*m


    // calc for the second time
    Mat newinput,newoutput;
    Mat outt=m_output_vec.t();
    Mat checkoutput = m_transform_mat*m_input_vec.t();
    checkoutput -= m_output_vec;
    int si = m_input_vec.rows;
    for(int k=0;k<si;k++)
    {
        if(checkoutput.col(k).dot(checkoutput.col(k))<2)
        {
            newinput.push_back(m_input_vec.row(k));
            newoutput.push_back(outt.row(k));
        }
    }

    cout<<"After selection: "<<newinput.rows<<endl;
    if(!newinput.rows)
    {
        return;
    }
    m_input_vec = newinput;
    m_output_vec = newoutput.t();

    result = (m_input_vec.t()*m_input_vec).inv(DECOMP_CHOLESKY)*m_input_vec.t()*m_output_vec.t(); //m * k
    m_transform_mat = result.t();// k*m

}
Mat LinearTransformManager::getTransformMat()
{
    return m_transform_mat;
}




#include "my_algorithm.h"
#include <sys/time.h>
#include <omp.h>
#include "parameters.h"
#include <opencv2/gpu/gpu.hpp>
#include <ostream>
using namespace cv;
using std::vector;
using std::cout;
using std::endl;
namespace
{
RNG rng(12345);
static const bool c_use_prydown = false;
static const bool c_down_twice = false;
static const bool c_display_chromatic_delta = false;

static const bool use_gauss = true;
static const int gauss_kernel_size = 5;
static const float gauss_sigma = 3;
//static const cv::Size frame_size = cv::Size(640,360);
//static const cv::Size frame_size = cv::Size(480,270);
static const cv::Size frame_size = cv::Size(320,180);
//static const cv::Size frame_size = cv::Size(240,135);
//static const cv::Size frame_size = cv::Size(160,90);
//static const cv::Size frame_size = cv::Size(0,0);
static const float resize_scale = 1/2.5;
static const int feature_points_num = 500;

static std::ofstream time_log("time.log");
static std::ofstream delta_frame_log("delta_sum.log");
static int cycle = 1;

static int mode =  1; // 1 normal; 2 clc sum delta frame;

// record time cost
struct timeval test_clock[2];
double getTimeCost()
{

    gettimeofday(&test_clock[1], NULL);

    double timeuse = 1000000 * (test_clock[1].tv_sec - test_clock[0].tv_sec)
            + test_clock[1].tv_usec - test_clock[0].tv_usec;
    timeuse /= 1000000;
    gettimeofday(&test_clock[0], NULL);
    return timeuse;
}
}

MyAlgorithm::MyAlgorithm():m_km_dataPts(2, particle_filter::c_particle_num)		// allocate data storage
{

}
MyAlgorithm::~MyAlgorithm()
{

}
Mat MyAlgorithm::getCurFrame()
{
    Mat retv = m_raw_frame.clone();
    if(m_min_std_error<30)
    {
        circle(retv,m_min_mid_p,m_min_std_error,Scalar(255,255,0),3);
    }
    return retv;
}
Mat MyAlgorithm::getPreFrame()
{
    for( size_t i=0;i<m_pre_klt_points.size();i++)
    {
        line(m_prev_frame,m_pre_klt_points[i],m_cur_klt_points[i],Scalar(255,255,255),1,8,0);
    }

    vector<Particles> pv = m_particle_filter.getParticles();
    for(size_t i=0;i<pv.size();i++)
    {
        rectangle(m_prev_frame,
                  Point(std::max(pv.at(i).m_col-1,0.0),std::max(pv.at(i).m_row-1,0.0)),
                  Point(pv.at(i).m_col,pv.at(i).m_row),Scalar(255,0,0));
    }

    for(size_t i=0;i<m_mid_p.size();i++)
    {
        circle(m_prev_frame,m_mid_p.at(i),m_std_error.at(i),Scalar(255,255,0),1);
    }

    return m_prev_frame;
}
Mat MyAlgorithm::getDeltaFrame()
{
    return m_delta_frame;
}
void MyAlgorithm::setVideo(VideoCapture *video)
{
    m_video_ptr = video;
}

void MyAlgorithm::initProcess()
{
    cout<<"GPU valid num: "<<gpu::getCudaEnabledDeviceCount()<<endl;
    if(m_video_ptr==NULL)
    {
        cout<<"No Video Input !!!"<<endl;
    }
    if( m_video_ptr->read(m_current_frame) && m_video_ptr->read(m_current_frame))
    {
//        for(int i=0;i<400;i++)
//        {
//            m_video_ptr->grab();
//        }
        m_video_ptr->retrieve(m_current_frame);
        cout<<"Video Process Start !!!"<<endl;
        if(c_use_prydown)
        {
            pyrDown(m_current_frame,m_current_frame,Size(m_current_frame.cols/2,m_current_frame.rows/2));
            if(c_down_twice)
            {
                pyrDown(m_current_frame,m_current_frame,Size(m_current_frame.cols/2,m_current_frame.rows/2));
            }
            m_klt_detector.setCurrentFrame(m_current_frame);
        }
        else
        {
            GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),gauss_sigma);
            cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
            if(c_down_twice)
            {
                GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),10);
                cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
            }
            m_klt_detector.setCurrentFrame(m_current_frame);
        }
    }
    else
    {
        std::cout<<"Error in video show !!!"<<std::endl;
    }

    m_particle_filter.initFilter(m_current_frame.rows,m_current_frame.cols);
}
void MyAlgorithm::doKLT()
{

    /// tracking and get pre frame
    m_klt_detector.doTrackingKLT();
    m_cur_klt_points = m_klt_detector.getCurKLTPoints();
    m_pre_klt_points = m_klt_detector.getPreKLTPoints();
}
void MyAlgorithm::doTransform()
{
    /// get transform matrix
    Mat input_x(m_cur_klt_points.size(),TRANSFORM_NUM,CV_64F);
    Mat input_y(m_cur_klt_points.size(),TRANSFORM_NUM,CV_64F);
    Mat output_x(1,m_pre_klt_points.size(),CV_64F);
    Mat output_y(1,m_pre_klt_points.size(),CV_64F);
    for(size_t t=0;t<m_pre_klt_points.size();t++)
    {

        input_y.at<double>(t,0) = input_x.at<double>(t,0) = m_cur_klt_points.at(t).x;
        input_y.at<double>(t,1) = input_x.at<double>(t,1) = m_cur_klt_points.at(t).y;
        input_y.at<double>(t,2) = input_x.at<double>(t,2) = 1;
        if(TRANSFORM_NUM==3)
        {

        }
        else if(TRANSFORM_NUM == 4)
        {
            input_y.at<double>(t,3) = input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;
        }
        else if(TRANSFORM_NUM == 5)
        {
            input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).x;
            input_x.at<double>(t,4) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;

            input_y.at<double>(t,3) = m_cur_klt_points.at(t).y*m_cur_klt_points.at(t).y;
            input_y.at<double>(t,4) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;

        }
        else if(TRANSFORM_NUM == 6)
        {
            input_y.at<double>(t,3) =input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).x;
            input_y.at<double>(t,4) =input_x.at<double>(t,4) = m_cur_klt_points.at(t).y*m_cur_klt_points.at(t).y;
            input_y.at<double>(t,5) =input_x.at<double>(t,5) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;
        }
        output_x.at<double>(0,t) = m_pre_klt_points.at(t).x;
        output_y.at<double>(0,t) = m_pre_klt_points.at(t).y;
    }
    m_lt_manager.setTransformInfo(input_x,input_y,output_x,output_y);
    m_lt_manager.solveTransform();
    time_log<<getTimeCost()<<" ";

    /// get delta frame
    if(c_display_chromatic_delta)
    {
        m_delta_frame = Mat(m_current_frame.rows,m_current_frame.cols,CV_8UC3);
    }
    else
    {
        m_delta_frame = Mat(m_current_frame.rows,m_current_frame.cols,CV_8UC1);
    }

#pragma omp parallel num_threads(THNUM)
    {
        Mat inpx(TRANSFORM_NUM,1,CV_64F),inpy(TRANSFORM_NUM,1,CV_64F);;
        int thr_id=omp_get_thread_num(); // get thread id
        int mini=m_delta_frame.rows/THNUM*thr_id;
        int maxi=mini+m_delta_frame.rows/THNUM;
        if(thr_id == THNUM-1)
        {
            maxi=std::max(m_delta_frame.rows,maxi);
        }

        Mat output_p;
        int a[2];

        for(size_t i=mini;i<maxi;i++)
        {
            for(size_t j=0;j<m_delta_frame.cols;j++)
            {
                inpy.at<double>(0,0)=inpx.at<double>(0,0)=j;//x,cols
                inpy.at<double>(1,0)=inpx.at<double>(1,0)=i;//y,rows
                inpy.at<double>(2,0)=inpx.at<double>(2,0)=1;
                if(TRANSFORM_NUM == 3)
                {

                }
                else if(TRANSFORM_NUM == 4)
                {
                    inpy.at<double>(3,0)=inpx.at<double>(3,0)=j*i;
                }
                else if(TRANSFORM_NUM ==5)
                {
                    inpx.at<double>(3,0)=j*j;
                    inpx.at<double>(4,0)=j*i;

                    inpy.at<double>(3,0)=i*i;
                    inpy.at<double>(4,0)=j*i;
                }
                else if(TRANSFORM_NUM == 6)
                {
                    inpy.at<double>(3,0)=inpx.at<double>(3,0)=j*j;
                    inpy.at<double>(4,0)=inpx.at<double>(4,0)=i*i;
                    inpy.at<double>(5,0)=inpx.at<double>(5,0)=j*i;
                }

                output_p = m_lt_manager.getOutputFromTransform(inpx,true); // output x
                a[0]=output_p.at<double>(0,0);//x,cols
                a[0]=min(a[0],m_delta_frame.cols-1);
                a[0]=max(a[0],0);

                output_p = m_lt_manager.getOutputFromTransform(inpy,false); // output y
                a[1]=output_p.at<double>(0,0);//y,rows
                a[1]=min(a[1],m_delta_frame.rows-1);
                a[1]=max(a[1],0);


//                a[0]=j;
//                a[1]=i;
                if(a[0]==0||a[0]==m_delta_frame.cols-1
                        ||a[1]==0||a[1]==m_delta_frame.rows-1)
                {
                    if(c_display_chromatic_delta)
                    {

                        m_delta_frame.at<Vec3b>(i,j)=Vec3b(0,0,0) ;
                    }
                    else
                    {
                        m_delta_frame.at<uchar>(i,j)=0 ;
                    }
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
                        hehe=max(hehe,(int)dd[k]);
                    }
                    if(c_display_chromatic_delta)
                    {
                        m_delta_frame.at<Vec3b>(i,j)=dd;
                    }
                    else
                    {
                        m_delta_frame.at<uchar>(i,j)=hehe;
                    }

                }
            }
        }
    }
    time_log<<getTimeCost()<<" ";
}
void MyAlgorithm::doTransform(int fnum)
{
    /// get transform matrix
    Mat input_x(m_cur_klt_points.size(),fnum,CV_64F);
    Mat input_y(m_cur_klt_points.size(),fnum,CV_64F);
    Mat output_x(1,m_pre_klt_points.size(),CV_64F);
    Mat output_y(1,m_pre_klt_points.size(),CV_64F);
    for(size_t t=0;t<m_pre_klt_points.size();t++)
    {

        input_y.at<double>(t,0) = input_x.at<double>(t,0) = m_cur_klt_points.at(t).x;
        input_y.at<double>(t,1) = input_x.at<double>(t,1) = m_cur_klt_points.at(t).y;
        input_y.at<double>(t,2) = input_x.at<double>(t,2) = 1;
        if(fnum==3)
        {

        }
        else if(fnum == 4)
        {
            input_y.at<double>(t,3) = input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;
        }
        else if(fnum == 5)
        {
            input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).x;
            input_x.at<double>(t,4) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;

            input_y.at<double>(t,3) = m_cur_klt_points.at(t).y*m_cur_klt_points.at(t).y;
            input_y.at<double>(t,4) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;

        }
        else if(fnum == 6)
        {
            input_y.at<double>(t,3) =input_x.at<double>(t,3) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).x;
            input_y.at<double>(t,4) =input_x.at<double>(t,4) = m_cur_klt_points.at(t).y*m_cur_klt_points.at(t).y;
            input_y.at<double>(t,5) =input_x.at<double>(t,5) = m_cur_klt_points.at(t).x*m_cur_klt_points.at(t).y;
        }
        output_x.at<double>(0,t) = m_pre_klt_points.at(t).x;
        output_y.at<double>(0,t) = m_pre_klt_points.at(t).y;
    }
    m_lt_manager.setTransformInfo(input_x,input_y,output_x,output_y);
    m_lt_manager.solveTransform();

    /// get delta frame
    if(c_display_chromatic_delta)
    {
        m_delta_frame = Mat(m_current_frame.rows,m_current_frame.cols,CV_8UC3);
    }
    else
    {
        m_delta_frame = Mat(m_current_frame.rows,m_current_frame.cols,CV_8UC1);
    }

#pragma omp parallel num_threads(THNUM)
    {
        Mat inpx(fnum,1,CV_64F),inpy(fnum,1,CV_64F);;
        int thr_id=omp_get_thread_num(); // get thread id
        int mini=m_delta_frame.rows/THNUM*thr_id;
        int maxi=mini+m_delta_frame.rows/THNUM;
        if(thr_id == THNUM-1)
        {
            maxi=std::max(m_delta_frame.rows,maxi);
        }

        Mat output_p;
        int a[2];

        for(size_t i=mini;i<maxi;i++)
        {
            for(size_t j=0;j<m_delta_frame.cols;j++)
            {
                inpy.at<double>(0,0)=inpx.at<double>(0,0)=j;//x,cols
                inpy.at<double>(1,0)=inpx.at<double>(1,0)=i;//y,rows
                inpy.at<double>(2,0)=inpx.at<double>(2,0)=1;
                if(fnum == 3)
                {

                }
                else if(fnum == 4)
                {
                    inpy.at<double>(3,0)=inpx.at<double>(3,0)=j*i;
                }
                else if(fnum ==5)
                {
                    inpx.at<double>(3,0)=j*j;
                    inpx.at<double>(4,0)=j*i;

                    inpy.at<double>(3,0)=i*i;
                    inpy.at<double>(4,0)=j*i;
                }
                else if(fnum == 6)
                {
                    inpy.at<double>(3,0)=inpx.at<double>(3,0)=j*j;
                    inpy.at<double>(4,0)=inpx.at<double>(4,0)=i*i;
                    inpy.at<double>(5,0)=inpx.at<double>(5,0)=j*i;
                }

                output_p = m_lt_manager.getOutputFromTransform(inpx,true); // output x
                a[0]=output_p.at<double>(0,0);//x,cols
                a[0]=min(a[0],m_delta_frame.cols-1);
                a[0]=max(a[0],0);

                output_p = m_lt_manager.getOutputFromTransform(inpy,false); // output y
                a[1]=output_p.at<double>(0,0);//y,rows
                a[1]=min(a[1],m_delta_frame.rows-1);
                a[1]=max(a[1],0);


                if(a[0]==0||a[0]==m_delta_frame.cols-1
                        ||a[1]==0||a[1]==m_delta_frame.rows-1)
                {
                    if(c_display_chromatic_delta)
                    {

                        m_delta_frame.at<Vec3b>(i,j)=Vec3b(0,0,0) ;
                    }
                    else
                    {
                        m_delta_frame.at<uchar>(i,j)=0 ;
                    }
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
                        hehe=max(hehe,(int)dd[k]);
                    }
                    if(c_display_chromatic_delta)
                    {
                        m_delta_frame.at<Vec3b>(i,j)=dd;
                    }
                    else
                    {
                        m_delta_frame.at<uchar>(i,j)=hehe;
                    }

                }
            }
        }
    }
}

void MyAlgorithm::doPF()
{
    m_particle_filter.setDiffFrame(m_delta_frame);
    m_particle_filter.doFiltering();
}

void MyAlgorithm::doClustering()
{


    vector<Particles> pv = m_particle_filter.getParticles();
    m_km_dataPts.setNPts(pv.size());
    KMdataArray ptr = m_km_dataPts.getPts();
    for(size_t i=0;i<pv.size();i++)
    {
        ptr[i][0] = pv[i].m_col;
        ptr[i][1] = pv[i].m_row;
    }
    m_km_dataPts.buildKcTree();

    int cernter_num = 5;
    KMfilterCenters ctrs(cernter_num, m_km_dataPts);		// allocate centers
    KMcenterArray center_array = ctrs.getCtrPts();          // init center points
    //    if(m_mid_p.size()>0)
    //    {
    //        for(size_t i=0;i<cernter_num;i++)
    //        {
    //            center_array[i][0]=m_mid_p.at(0).x;
    //            center_array[i][1]=m_mid_p.at(0).y;
    //        }
    //    }


    KMterm	term(1000, 0, 0, 0,		// run for 1000 stages
                 0.10,			// min consec RDL
                 0.10,			// min accum RDL
                 200,			// max run stages
                 0.50,			// init. prob. of acceptance
                 10,			// temp. run length
                 0.95);			// temp. reduction factor

    // run the algorithms
    KMlocalLloyds kmLloyds(ctrs, term);		// repeated Lloyd's
    ctrs = kmLloyds.execute();			// execute

    // get centers & errors
    m_mid_p.clear();
    m_std_error.clear();
    m_min_std_error = DBL_MAX;

    double* dist = ctrs.getDists();
    int* weights = ctrs.getWeights();
    double std_error_temp;
    for(size_t i=0;i<cernter_num;i++)
    {
        m_mid_p.push_back(Point2f(center_array[i][0],center_array[i][1]));
        std_error_temp = std::sqrt(dist[i]/weights[i]);
        m_std_error.push_back(std_error_temp);
        if(std_error_temp<m_min_std_error)
        {
            m_min_std_error = std_error_temp;
            m_min_mid_p = Point2f(center_array[i][0],center_array[i][1]);
        }
    }
}
void MyAlgorithm::sumDeltaFrame()
{
    int row = m_delta_frame.rows;
    int col = m_delta_frame.cols;
    long sum=0;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            sum+=m_delta_frame.at<uchar>(i,j);
        }
    }
    delta_frame_log<<sum<<" ";
}


void MyAlgorithm::processNextFrame()
{
    getTimeCost();



    if(mode==1)
    {

        m_prev_frame = m_current_frame.clone();
        if(m_video_ptr==NULL)
        {
            cout<<"No Video Input !!!"<<endl;
        }
        /// get next frame as current frame
        if( !m_video_ptr->read(m_current_frame))//||!m_video_ptr->read(m_current_frame))
        {
            std::cout<<"Error in video show !!!"<<std::endl;
            return;
        }
        if(c_use_prydown)
        {
            pyrDown(m_current_frame,m_current_frame,Size(m_current_frame.cols/2,m_current_frame.rows/2));
            if(c_down_twice)
            {
                pyrDown(m_current_frame,m_current_frame,Size(m_current_frame.cols/2,m_current_frame.rows/2));
            }
            m_klt_detector.setCurrentFrame(m_current_frame);
        }
        else
        {
            cv::resize(m_current_frame,m_raw_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
            if(c_down_twice)
            {
                GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),10);
                cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
            }
            if(use_gauss)
            {
                GaussianBlur(m_raw_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),gauss_sigma);
            }
            else
            {
                m_current_frame = m_raw_frame;
            }
            m_klt_detector.setCurrentFrame(m_raw_frame);
        }



        time_log<<cycle++<<" ";
        doKLT();
        //    cout<<"Time KLT: "<< getTimeCost()<<"\n";
        doTransform();
        //    cout<<"Time Tra: "<< getTimeCost()<<"\n";
        doPF();
        time_log<<getTimeCost()<<" ";
        //    cout<<"Time PF : "<< getTimeCost()<<"\n";
        doClustering();
        time_log<<getTimeCost()<<" ";
        //    cout<<"Time Clu: "<< getTimeCost()<<"\n"<<endl;
        time_log<<endl;
    }
    else if(mode == 2)
    {
        for(int i=0;i<300;i++)
        {
            if(m_video_ptr==NULL)
            {
                cout<<"No Video Input !!!"<<endl;
            }
            /// get next frame as current frame
            if( !m_video_ptr->read(m_current_frame))//||!m_video_ptr->read(m_current_frame))
            {
                std::cout<<"Error in video show !!!"<<std::endl;
                return;
            }
            delta_frame_log<<cycle++<<" ";
            doKLT();
            for(int i=3;i<=6;i++)
            {
                doTransform(i);
                sumDeltaFrame();
            }
            delta_frame_log<<m_pre_klt_points.size()<<endl;
        }
    }
}




KLTDetector::KLTDetector()
{

}
void KLTDetector::setCurrentFrame(Mat frame)
{
    if(!USE_CUDA)
    {
        swap(m_prev_frame_gray,m_current_frame_gray);
        m_prev_frame = m_current_frame.clone();
        m_current_frame = frame;
        cvtColor(m_current_frame,m_current_frame_gray,CV_BGR2GRAY);
    }
    else
    {
        //        getTimeCost();
        m_prev_frame = m_current_frame.clone();
        m_current_frame = frame;
        //        cout<<"KLT 11: "<<getTimeCost()<<endl;
        swap(m_prev_frame_gray_gpu,m_current_frame_gray_gpu);
        swap(m_prev_frame_gpu,m_current_frame_gpu);
        m_current_frame_gpu = gpu::GpuMat(frame);
        //        cout<<"KLT 12: "<<getTimeCost()<<endl;
        cvtColor(m_current_frame_gpu,m_current_frame_gray_gpu,CV_BGR2GRAY);
        //        cout<<"KLT 13: "<<getTimeCost()<<endl;
    }
}
void KLTDetector::doTrackingKLT()
{
    vector<Point2f> prev_points;
    vector<Point2f> cur_points;
    if(!USE_CUDA)
    {
        // get good feature points
        m_pre_klt_points = get_KLT_features(m_prev_frame_gray,feature_points_num);
        time_log<<getTimeCost()<<" ";

        // get optical flow
        std::vector<unsigned char> features_status;
        std::vector<float> features_error;
        Size search_window(21,21);
        int pyr_level = 6;
        calcOpticalFlowPyrLK(m_prev_frame, m_current_frame,
                             m_pre_klt_points, m_cur_klt_points, features_status, features_error,
                             search_window, pyr_level,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                             OPTFLOW_LK_GET_MIN_EIGENVALS,1e-4);

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
        time_log<<getTimeCost()<<" ";
    }
    else
    {
        //detect first
        double minDistBetweenPoints = 0;
        double minLevel = 0.07;
        gpu::GoodFeaturesToTrackDetector_GPU detector(feature_points_num, minLevel, minDistBetweenPoints);
        gpu::GpuMat prevP,newP,status,error;
        detector(m_prev_frame_gray_gpu,prevP);
        cout<<"start prevP: "<<prevP.cols<<endl;
        time_log<<getTimeCost()<<" ";

        // get flow in gpumat
        gpu::PyrLKOpticalFlow d_pyrLK;
        d_pyrLK.winSize.width = 21;
        d_pyrLK.winSize.height = 21;
        d_pyrLK.maxLevel = 7;
        d_pyrLK.iters = 30;
        d_pyrLK.sparse(m_prev_frame_gray_gpu,m_current_frame_gray_gpu,
                       prevP,newP,status,&error);

        // convert gpumat 2 vector<>
        prev_points.resize(prevP.cols);
        cur_points.resize(newP.cols);
        Mat prevM(1,prevP.cols,CV_32FC2,(void*)&prev_points[0]);
        prevP.download(prevM);
        Mat currM(1,newP.cols,CV_32FC2,(void*)&cur_points[0]);
        newP.download(currM);

        vector<uchar> statusV(newP.cols);
        Mat statusM(1,newP.cols,CV_8UC1,(void*)&statusV[0]);
        status.download(statusM);

        vector<float> errorV(newP.cols);
        Mat errorM(1,newP.cols,CV_32FC1,(void*)&errorV[0]);
        error.download(errorM);

        // delete bad points
        m_pre_klt_points.clear();
        m_cur_klt_points.clear();
        for( size_t i=0;i<prev_points.size();i++)
        {
            if( !statusV[i] || errorV[i]>20 )
            {
                continue;
            }
            m_pre_klt_points.push_back(prev_points.at(i));
            m_cur_klt_points.push_back(cur_points.at(i));
        }
        cout<<"end newP: "<<m_pre_klt_points.size()<<endl;
        time_log<<getTimeCost()<<" ";

    }
}
std::vector<Point2f> KLTDetector::get_KLT_features(Mat image,int maxCorners)
{
    Mat in_image;
    if(image.channels()==3)
    {
        cvtColor(image,in_image,CV_BGR2GRAY);
    }
    else
    {
        in_image=image.clone();
    }


    // Parameters for Shi-Tomasi algorithm
    vector<Point2f> corners;
    double qualityLevel = 0.05;
    double minDistance = 5;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.01;

    // Apply corner detection
    goodFeaturesToTrack( in_image,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         useHarrisDetector,
                         k );


    // find corner sub-pixel
    cornerSubPix(in_image, corners, Size(blockSize,blockSize), Size(-1,-1),
                 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,20,0.03));


    // output corners number
    //    std::cout<<"** Number of corners detected: "<<corners.size()<<std::endl;
    return corners;
}
const std::vector<cv::Point2f>& KLTDetector::getPreKLTPoints()
{
    return m_pre_klt_points;
}
const std::vector<cv::Point2f>& KLTDetector::getCurKLTPoints()
{
    return m_cur_klt_points;
}
cv::Mat KLTDetector::getPreFrame()
{
    if(!USE_CUDA)
    {
        return m_prev_frame;
    }
    else
    {
        return m_prev_frame;
    }

}
cv::Mat KLTDetector::getCurFrame()
{
    return m_current_frame;
}






LinearTransformManager::LinearTransformManager()
{

}
void LinearTransformManager::setTransformInfo(cv::Mat input_vec_x,cv::Mat input_vec_y,cv::Mat output_vec_x,cv::Mat output_vec_y)
{
    m_input_vec_x = input_vec_x;
    m_input_vec_y = input_vec_y;
    m_output_vec_x = output_vec_x;
    m_output_vec_y = output_vec_y;
}
void LinearTransformManager::solveTransform()
{
    Mat result_x;
    result_x = (m_input_vec_x.t()*m_input_vec_x).inv(DECOMP_CHOLESKY)*m_input_vec_x.t()*m_output_vec_x.t(); //m * k
    m_transform_mat_x = result_x.t();// k*m

    Mat result_y;
    result_y = (m_input_vec_y.t()*m_input_vec_y).inv(DECOMP_CHOLESKY)*m_input_vec_y.t()*m_output_vec_y.t(); //m * k
    m_transform_mat_y = result_y.t();// k*m

    // calc for the second time
    Mat newinput_x,newoutput_x,newinput_y,newoutput_y;
    Mat deltax,deltay;
    deltax = m_output_vec_x-m_transform_mat_x*m_input_vec_x.t();
    deltay = m_output_vec_y-m_transform_mat_y*m_input_vec_y.t();
    Mat outxt=m_output_vec_x.t();
    Mat outyt=m_output_vec_y.t();
    //    Mat checkoutput = m_transform_mat_x*m_input_vec_x.t();
    //    checkoutput -= m_output_vec;
    int si = m_input_vec_x.rows;
    for(int k=0;k<si;k++)
    {
        if((deltax.at<double>(0,k)*deltax.at<double>(0,k)+deltay.at<double>(0,k)*deltay.at<double>(0,k)) <5)
        {
            newinput_x.push_back(m_input_vec_x.row(k));
            newinput_y.push_back(m_input_vec_y.row(k));
            newoutput_x.push_back(outxt.row(k));
            newoutput_y.push_back(outyt.row(k));
        }
    }

    //    cout<<"After selection: "<<newinput.rows<<endl;
    if(!newinput_x.rows)
    {
        return;
    }
    m_input_vec_x = newinput_x;
    m_output_vec_x = newoutput_x.t();
    m_input_vec_y = newinput_y;
    m_output_vec_y = newoutput_y.t();

    result_x = (m_input_vec_x.t()*m_input_vec_x).inv(DECOMP_CHOLESKY)*m_input_vec_x.t()*m_output_vec_x.t(); //m * k
    result_y = (m_input_vec_y.t()*m_input_vec_y).inv(DECOMP_CHOLESKY)*m_input_vec_y.t()*m_output_vec_y.t(); //m * k
    m_transform_mat_x = result_x.t();// k*m
    m_transform_mat_y = result_y.t();// k*m

}
Mat LinearTransformManager::getTransformMat()
{
    return m_transform_mat_x;
}




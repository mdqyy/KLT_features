#include "my_algorithm.h"
#include <sys/time.h>
#include <omp.h>
#include "parameters.h"
#include <opencv2/gpu/gpu.hpp>
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

static const int gauss_kernel_size = 3;
static const float gauss_sigma = 3;

static const cv::Size frame_size = cv::Size(320,180);
//static const cv::Size frame_size = cv::Size(0,0);
static const float resize_scale = 1/2.5;
static const int feature_points_num = 2000;
static const int features_num = 4;




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
    return m_current_frame;
}
Mat MyAlgorithm::getPreFrame()
{
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
        for(int i=0;i<100;i++)
        {
            m_video_ptr->grab();
        }
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
    getTimeCost();
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
        getTimeCost();
//        GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),gauss_sigma);
        cout<<"KLT 00: "<<getTimeCost()<<endl;
        cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
        if(c_down_twice)
        {
            GaussianBlur(m_current_frame,m_current_frame,Size(gauss_kernel_size,gauss_kernel_size),10);
            cv::resize(m_current_frame,m_current_frame,frame_size,resize_scale,resize_scale,INTER_LINEAR );
        }
//        cout<<"KLT 01: "<<getTimeCost()<<endl;
        m_klt_detector.setCurrentFrame(m_current_frame);
    }
//    cout<<"KLT 1: "<<getTimeCost();
    /// tracking and get pre frame
    m_klt_detector.doTrackingKLT();
//    cout<<"KLT 2: "<<getTimeCost();
    m_prev_frame = m_klt_detector.getPreFrame();
}
void MyAlgorithm::doTransform()
{
    vector<Point2f> cur_klt_points,pre_klt_points;
    cur_klt_points = m_klt_detector.getCurKLTPoints();
    pre_klt_points = m_klt_detector.getPreKLTPoints();


    /// get transform matrix
    Mat input(cur_klt_points.size(),features_num,CV_64F);
    Mat output(2,pre_klt_points.size(),CV_64F);
    for(size_t t=0;t<pre_klt_points.size();t++)
    {

        input.at<double>(t,0) = cur_klt_points.at(t).x;
        input.at<double>(t,1) = cur_klt_points.at(t).y;
        input.at<double>(t,2) = 1;
        if(features_num == 4)
        {
            input.at<double>(t,3) = cur_klt_points.at(t).x*cur_klt_points.at(t).y;
        }
        output.at<double>(0,t) = pre_klt_points.at(t).x;
        output.at<double>(1,t) = pre_klt_points.at(t).y;
    }
    m_lt_manager.setTransformInfo(input,output);
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



    getTimeCost();

#pragma omp parallel num_threads(THNUM)
    {
        Mat inp(features_num,1,CV_64F);
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
                inp.at<double>(0,0)=j;//x,cols
                inp.at<double>(1,0)=i;//y,rows
                inp.at<double>(2,0)=1;
                inp.at<double>(3,0)=j*i;
                //                Mat trans = m_lt_manager.getTransformMat();
                //                output_p = trans*inp;
                output_p = m_lt_manager.getOutputFromTransform(inp);
                a[0]=output_p.at<double>(0,0);//x,cols
                a[0]=min(a[0],m_delta_frame.cols-1);
                a[0]=max(a[0],0);
                a[1]=output_p.at<double>(1,0);//y,rows
                a[1]=min(a[1],m_delta_frame.rows-1);
                a[1]=max(a[1],1);
                //            cout<<"Time Tra3: "<< getTimeCost()<<"\n";

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

    for( size_t i=0;i<pre_klt_points.size();i++)
    {
        line(m_prev_frame,pre_klt_points[i],cur_klt_points[i],Scalar(255,255,255),1,8,0);
    }

}


void MyAlgorithm::doPF()
{
    m_particle_filter.setDiffFrame(m_delta_frame);
    m_particle_filter.doFiltering();
    vector<Particles> pv = m_particle_filter.getParticles();
    for(size_t i=0;i<pv.size();i++)
    {
        rectangle(m_prev_frame,
                  Point(std::max(pv.at(i).m_col-2,0.0),std::max(pv.at(i).m_row-2,0.0)),
                  Point(pv.at(i).m_col,pv.at(i).m_row),Scalar(255,0,0));
    }
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
    KMterm	term(1000, 0, 0, 0,		// run for 100 stages
                 0.10,			// min consec RDL
                 0.10,			// min accum RDL
                 3,			// max run stages
                 0.50,			// init. prob. of acceptance
                 10,			// temp. run length
                 0.95);			// temp. reduction factor

    // run the algorithms
    //    cout << "\nExecuting Clustering Algorithm: Lloyd's\n";
    KMlocalLloyds kmLloyds(ctrs, term);		// repeated Lloyd's
    ctrs = kmLloyds.execute();			// execute

    // get centers & errors
    Point mid;
    double* dist = ctrs.getDists();
    int* weights = ctrs.getWeights();
    KMcenterArray center_array = ctrs.getCtrPts();
    int min_std_error_id=0;
    double min_std_error = DBL_MAX;
    double std_error_temp;
    for(size_t i=0;i<cernter_num;i++)
    {
        //        cout<<center_array[i][0]<<" "<<center_array[i][1]<<" "<<weights[i]<<" "<<std::sqrt(dist[i]/weights[i])<<endl;
        std_error_temp = std::sqrt(dist[i]/weights[i]);
        if(std_error_temp<min_std_error)
        {
            min_std_error = std_error_temp;
            min_std_error_id = i;
        }
        mid = Point(center_array[i][0],center_array[i][1]);
//        circle(m_prev_frame,mid,std_error_temp,Scalar(255,255,0),2);
    }

    //    cout<<min_std_error<<endl;
    if(min_std_error>30)
    {
        return;
    }
    mid = Point(center_array[min_std_error_id][0],center_array[min_std_error_id][1]);
    //    cout<<mid<<endl;
        cout<<min_std_error<<endl;
    circle(m_prev_frame,mid,min_std_error,Scalar(255,255,0),3);
}



void MyAlgorithm::processNextFrame()
{
    getTimeCost();

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


    doKLT();
    cout<<"Time KLT: "<< getTimeCost()<<"\n";
    doTransform();
    cout<<"Time Tra: "<< getTimeCost()<<"\n";
    doPF();
    cout<<"Time PF : "<< getTimeCost()<<"\n";
    doClustering();
    cout<<"Time Clu: "<< getTimeCost()<<"\n"<<endl;
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
        m_pre_klt_points = get_KLT_features(m_prev_frame_gray,feature_points_num);

        // get motion line
        std::vector<unsigned char> features_status;
        std::vector<float> features_error;
        Size search_window(21,21);
        int pyr_level = 6;
        calcOpticalFlowPyrLK(m_prev_frame, m_current_frame,
                             m_pre_klt_points, m_cur_klt_points, features_status, features_error,
                             search_window, pyr_level,
                             TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                             OPTFLOW_LK_GET_MIN_EIGENVALS,1e-4);
        //        std::vector<Point2f> prev_points,cur_points;
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
    }
    else
    {
        //detect first
        double minDistBetweenPoints = 0;
        double minLevel = 0.05;
        gpu::GoodFeaturesToTrackDetector_GPU detector(feature_points_num, minLevel, minDistBetweenPoints);
        gpu::GpuMat prevP,newP,status,error;
        detector(m_prev_frame_gray_gpu,prevP);
        cout<<"start prevP: "<<prevP.cols<<endl;

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
    if(!USE_CUDA)
    {
        return m_prev_frame;
    }
    else
    {
        return m_prev_frame;
//        Mat retv;
//        m_prev_frame_gray_gpu.download(retv);
//        retv.convertTo(retv,CV_8UC1,);
//        return Mat(m_prev_frame_gray_gpu);
    }

}
cv::Mat KLTDetector::getCurFrame()
{
    return m_current_frame;
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
        if(checkoutput.col(k).dot(checkoutput.col(k))<5)
        {
            newinput.push_back(m_input_vec.row(k));
            newoutput.push_back(outt.row(k));
        }
    }

    //    cout<<"After selection: "<<newinput.rows<<endl;
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




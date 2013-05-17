#include "particle_filter.h"
#include <algorithm>
#include <omp.h>
#include <sys/time.h>
#include "parameters.h"

using cv::Mat;
using std::vector;
using std::cout;
using std::flush;
using std::endl;
namespace particle_filter
{
static const int c_pro_mask_width = 5;
static const int c_time = 1.0/15;
int c_particle_num = 2000;
static const double c_gaussion_variance = 60;
static const double c_gaussion_variance_v = 225;
double utils_gaussrand(double expectation,double variance,int thread_id=0)
{
    static int phase[THNUM] = {0};
    double X;
    static double S[THNUM]={0};
    static double V1[THNUM]={0}, V2[THNUM]={0};

    if ( phase[thread_id] == 0 )
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1[thread_id] = 2 * U1 - 1;
            V2[thread_id] = 2 * U2 - 1;
            S[thread_id] = V1[thread_id] * V1[thread_id] + V2[thread_id] * V2[thread_id];
        }
        while(S[thread_id] >= 1 || S[thread_id] == 0);

        X = V1[thread_id] * sqrt(-2 * log(S[thread_id]) / S[thread_id]);
    }
    else
        X = V2[thread_id] * sqrt(-2 * log(S[thread_id]) / S[thread_id]);


    phase[thread_id] = 1 - phase[thread_id];

    double d = (variance>0)?sqrt(variance):sqrt(-variance);

    return X*d + expectation;
}

double utils_randBetween(int begin,int end)
{
    int range = abs(end-begin)+1;
    int rand_val = rand()%range;
    if(begin<end)
        return begin + rand_val;
    else
        return end + rand_val;
}

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

using namespace particle_filter;



ParticleFilter::ParticleFilter()
{
    m_rows=0;
    m_cols=0;
}

void ParticleFilter::setDiffFrame(cv::Mat diff)
{
    m_frame_distribution.create(m_rows,m_cols,CV_64F);
    double max=0;
    double sum = 0;
    for(int i=0;i<m_rows;i++)
    {
        for(int j=0;j<m_cols;j++)
        {
            sum += diff.at<unsigned char>(i,j);
            if(max<diff.at<unsigned char>(i,j))
            {
                max = diff.at<unsigned char>(i,j);
            }
        }
    }
//    cout<<max<<endl;
    for(int i=0;i<m_rows;i++)
    {
        for(int j=0;j<m_cols;j++)
        {
            m_frame_distribution.at<double>(i,j) = diff.at<unsigned char>(i,j)*1.0/sum;
        }
    }
}
void ParticleFilter::doFiltering()
{
    motionStep();
    measureStep();
    resampleStep();
}

const vector<Particles>& ParticleFilter::getParticles()
{
    return m_particles;
}

void ParticleFilter::initFilter(int rows,int cols)
{
    m_rows = rows;
    m_cols = cols;
    m_particles.clear();
    Particles tempp;
    tempp.m_weight = 1.0/c_particle_num;
    for(int i=0;i<c_particle_num;i++)
    {
        tempp.m_row = utils_randBetween(0,m_rows);
        tempp.m_col = utils_randBetween(0,m_cols);
        tempp.m_row_v = utils_gaussrand(0,c_gaussion_variance_v);
        tempp.m_col_v = utils_gaussrand(0,c_gaussion_variance_v);
        m_particles.push_back(tempp);
    }
}
void ParticleFilter::motionStep()
{
    int le = c_pro_mask_width/2;
    int thrid=0;
    for(size_t i=0;i<m_particles.size();i++)
    {
        m_particles[i].m_row += c_time*m_particles[i].m_row_v+utils_gaussrand(0,c_gaussion_variance,thrid);
        m_particles[i].m_row = m_particles[i].m_row>(m_rows-1-le)?(m_rows-1-le):m_particles[i].m_row;
        m_particles[i].m_row = m_particles[i].m_row<(le)?(le):m_particles[i].m_row;
        m_particles[i].m_col +=c_time* m_particles[i].m_col_v+utils_gaussrand(0,c_gaussion_variance,thrid);
        m_particles[i].m_col = m_particles[i].m_col>(m_cols-1-le)?(m_cols-1-le):m_particles[i].m_col;
        m_particles[i].m_col = m_particles[i].m_col<(le)?(le):m_particles[i].m_col;
        m_particles[i].m_row_v += utils_gaussrand(0,c_gaussion_variance_v,thrid);
        m_particles[i].m_col_v += utils_gaussrand(0,c_gaussion_variance_v,thrid);
    }
}
void ParticleFilter::measureStep()
{
    double sum;
    for(size_t i=0;i<m_particles.size();i++)
    {
        m_particles[i].m_weight *= getWeight(m_particles.at(i).m_row,m_particles.at(i).m_col);
        sum+=m_particles[i].m_weight;
    }
    for(size_t i=0;i<m_particles.size();i++)
    {
        m_particles[i].m_weight /= sum;
    }
}
double ParticleFilter::getWeight(int rows, int cols)
{
    //    cout<<"2.1"<<flush;
    int le = c_pro_mask_width/2;
    double retv = 1e-6;//avoid return zero
    for(size_t j=rows-le;j<rows+le+1;j++)
    {
        for(size_t i=cols-le;i<cols+le+1;i++)
        {
            //            cout<<" "<<i<<" "<<j<<endl;
            retv+=m_frame_distribution.at<double>(j,i);
        }
    }
    //    cout<<"2.2"<<flush;
    return retv;
}
void ParticleFilter::resampleStep()
{

    vector<Particles> new_part;
    Particles tempp;

    double t_c[c_particle_num];
    t_c[0] = m_particles[0].m_weight;
    for(int i=1; i<c_particle_num; i++)
    {
        t_c[i] = t_c[i-1] + m_particles[i].m_weight;
    }

    int i=0;
    double t_u[c_particle_num];
    int le = c_pro_mask_width/2;
    t_u[0] = rand()*1.0/RAND_MAX/c_particle_num;
    for(int j=0; j<m_particles.size(); j++)
    {
        t_u[j] = t_u[0] + 1.0/c_particle_num* j;
        while(t_u[j] > t_c[i])
        {
            i++;
        }
        if((rand()*1.0/RAND_MAX)<0.25)
        {
            tempp.m_row = utils_randBetween(0,m_rows);
            tempp.m_col = utils_randBetween(0,m_cols);
            tempp.m_row_v = utils_gaussrand(0,c_gaussion_variance_v);
            tempp.m_col_v = utils_gaussrand(0,c_gaussion_variance_v);
            tempp.m_weight = 1.0/c_particle_num;
        }
        else
        {

            tempp.m_row = m_particles.at(i).m_row+utils_gaussrand(0,c_gaussion_variance);
            tempp.m_col = m_particles.at(i).m_col+utils_gaussrand(0,c_gaussion_variance);
            tempp.m_row_v = m_particles.at(i).m_row_v+utils_gaussrand(0,c_gaussion_variance_v);
            tempp.m_col_v = m_particles.at(i).m_col_v+utils_gaussrand(0,c_gaussion_variance_v);
            tempp.m_weight = 1.0/c_particle_num;
        }
        new_part.push_back(tempp);
    }

    m_particles = new_part;
}

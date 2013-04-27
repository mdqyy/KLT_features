#include "particle_filter.h"
#include <algorithm>
using cv::Mat;
using std::vector;
using std::cout;
using std::flush;
using std::endl;
namespace
{
static const int c_pro_mask_width = 7;
static const int c_time = 1.0/15;
static const int c_particle_num = 1000;

double utils_gaussrand(double expectation,double variance)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 )
    {
        do
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        }
        while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

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
}

ParticleFilter::ParticleFilter()
{
    m_rows=0;
    m_cols=0;
}

void ParticleFilter::setDiffFrame(cv::Mat diff)
{
    m_frame_distribution.create(m_rows,m_cols,CV_64F);

    double sum = 0;
    for(int i=0;i<m_rows;i++)
    {
        for(int j=0;j<m_cols;j++)
        {
            sum += diff.at<unsigned char>(i,j);
        }
    }
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
    cout<<"1"<<std::flush;
    motionStep();
    cout<<"2"<<std::flush;
    measureStep();
    cout<<"3"<<std::flush;
    resampleStep();
    cout<<"4"<<std::endl;

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
        tempp.m_row_v = utils_gaussrand(0,4);
        tempp.m_col_v = utils_gaussrand(0,4);
        m_particles.push_back(tempp);
    }
}
void ParticleFilter::motionStep()
{
    int le = c_pro_mask_width/2;
    for(size_t i=0;i<m_particles.size();i++)
    {
        m_particles[i].m_row += c_time*m_particles[i].m_row_v+utils_gaussrand(0,4);
        m_particles[i].m_row = m_particles[i].m_row>(m_rows-1-le)?(m_rows-1-le):m_particles[i].m_row;
        m_particles[i].m_row = m_particles[i].m_row<(le)?(le):m_particles[i].m_row;
        m_particles[i].m_col +=c_time* m_particles[i].m_col_v+utils_gaussrand(0,4);
        m_particles[i].m_col = m_particles[i].m_col>(m_cols-1-le)?(m_cols-1-le):m_particles[i].m_col;
        m_particles[i].m_col = m_particles[i].m_col<(le)?(le):m_particles[i].m_col;
        m_particles[i].m_row_v += utils_gaussrand(0,4);
        m_particles[i].m_col_v += utils_gaussrand(0,4);
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

    int i;
    double t_u[c_particle_num];
    int le = c_pro_mask_width/2;
    t_u[0] = rand()*1.0/RAND_MAX/c_particle_num;
    for(int j=0; j<m_particles.size(); j++)
    {
        t_u[j] = t_u[0] + 1.0/c_particle_num* j;
        i=0;
        while(t_u[j] > t_c[i])
        {
            i++;
        }
        //        cout<<i<<endl;
        if((rand()*1.0/RAND_MAX)<0.25)
        {
            tempp.m_row = utils_randBetween(0,m_rows);
            tempp.m_col = utils_randBetween(0,m_cols);
            tempp.m_row_v = utils_gaussrand(0,4);
            tempp.m_col_v = utils_gaussrand(0,4);
            tempp.m_weight = 1.0/c_particle_num;
        }
        else
        {

            tempp.m_row = m_particles.at(i).m_row+utils_gaussrand(0,4);
            tempp.m_col = m_particles.at(i).m_col+utils_gaussrand(0,4);
            tempp.m_row_v = m_particles.at(i).m_row_v+utils_gaussrand(0,4);
            tempp.m_col_v = m_particles.at(i).m_col_v+utils_gaussrand(0,4);
            tempp.m_weight = 1.0/c_particle_num;
        }
        new_part.push_back(tempp);
    }

    m_particles = new_part;
}

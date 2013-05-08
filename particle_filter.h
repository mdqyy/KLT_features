#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H
#include <opencv2/opencv.hpp>
#include <vector>

namespace particle_filter
{
extern int c_particle_num;
}

struct Particles
{
    Particles()
    {
        m_row = (0);
        m_col = (0);
        m_row_v = (0);
        m_col_v = (0);
        m_weight = (1);
    }


    double m_row;
    double m_col;
    double m_row_v;
    double m_col_v;
    double m_weight;
};

class ParticleFilter
{
public:
    ParticleFilter();
    void initFilter(int rows,int cols);
    void setDiffFrame(cv::Mat diff);
    void doFiltering();
    const std::vector<Particles>& getParticles();
private:
    double getWeight(int rows,int cols);
    void motionStep();
    void measureStep();
    void resampleStep();
private:
    cv::Mat m_frame_distribution;
    std::vector<Particles> m_particles;
    int m_rows;
    int m_cols;
};

#endif // PARTICLE_FILTER_H

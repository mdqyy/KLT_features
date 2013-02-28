#ifndef MAIN_WIDGET_H
#define MAIN_WIDGET_H

#include <QtGui>
#include <opencv2/opencv.hpp>
#include <vector>

using std::vector;

namespace Ui {
class MainWidget;
}

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = 0);
    ~MainWidget();
    void showImage(cv::Mat a);
    void showImage(cv::Mat a,QLabel* label);


    void showVideo(cv::VideoCapture& video);
    void loadImage(QString s);
    void loadVideo(QString s);
    void keyPressEvent(QKeyEvent *);
    void processVideo(cv::VideoCapture& video);
    vector < cv::Point2f > get_KLT_features(cv::Mat image, int maxCorners);

private slots:
    void processImage();
private:
    Ui::MainWidget *ui;
    cv::Mat m_current_frame;
    cv::Mat m_current_frame_gray;
    cv::Mat m_prev_frame;
    cv::Mat m_prev_frame_gray;
    cv::Mat m_show_frame;
    cv::VideoCapture m_video;
    int m_cycle;
};

#endif // MAIN_WIDGET_H

#-------------------------------------------------
#
# Project created by QtCreator 2013-02-27T10:45:47
#
#-------------------------------------------------

QT       += core gui\

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = klt_feature

TEMPLATE = app


SOURCES += main.cpp\
        main_widget.cpp \
    my_algorithm.cpp

HEADERS  += main_widget.h \
    my_algorithm.h

FORMS    += main_widget.ui

win32 {
INCLUDEPATH += E:\opencv_debug\install\include \
#INCLUDEPATH += D:\OpenCV\opencv\debug\install\include \

LIBS +=  \
 E:\opencv_debug\install\lib\libopencv_highgui245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_stitching245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_videostab245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_gpu245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_ml245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_features2d245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_calib3d245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_objdetect245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_contrib245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_legacy245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_flann245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_video245d.dll.a\
 E:\opencv_debug\install\lib\libopencv_imgproc245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_core245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_nonfree245d.dll.a \
 E:\opencv_debug\install\lib\libopencv_photo245d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_highgui244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_stitching244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_videostab244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_gpu244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_ml244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_features2d244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_calib3d244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_objdetect244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_contrib244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_legacy244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_flann244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_video244d.dll.a\
# D:\OpenCV\opencv\debug\install\lib\libopencv_imgproc244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_core244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_nonfree244d.dll.a \
# D:\OpenCV\opencv\debug\install\lib\libopencv_photo244d.dll.a \
} else {

INCLUDEPATH += /usr/local/include/opencv-2.4.3/\

LIBS += -L/usr/local/lib -lopencv_core\
         -lopencv_imgproc\
 -lopencv_highgui\
 -lopencv_ml\
 -lopencv_video\
 -lopencv_features2d\
 -lopencv_calib3d\
 -lopencv_objdetect\
 -lopencv_contrib\
 -lopencv_legacy\
 -lopencv_flann

}



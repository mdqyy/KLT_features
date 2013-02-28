#-------------------------------------------------
#
# Project created by QtCreator 2013-02-27T10:45:47
#
#-------------------------------------------------

QT       += core gui

TARGET = klt_feature

TEMPLATE = app


SOURCES += main.cpp\
        main_widget.cpp

HEADERS  += main_widget.h

FORMS    += main_widget.ui

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

/********************************************************************************
** Form generated from reading UI file 'main_widget.ui'
**
** Created: Thu Apr 4 11:56:31 2013
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAIN_WIDGET_H
#define UI_MAIN_WIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWidget
{
public:
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout_2;
    QLabel *m_main_label;
    QLabel *m_main_label_2;
    QVBoxLayout *verticalLayout;
    QPushButton *m_button_start;
    QPushButton *m_button_open;

    void setupUi(QWidget *MainWidget)
    {
        if (MainWidget->objectName().isEmpty())
            MainWidget->setObjectName(QString::fromUtf8("MainWidget"));
        MainWidget->resize(720, 837);
        MainWidget->setMinimumSize(QSize(720, 720));
        MainWidget->setMaximumSize(QSize(9999, 9999));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icon/lena.ico"), QSize(), QIcon::Normal, QIcon::On);
        icon.addFile(QString::fromUtf8(":/icon/lena.ico"), QSize(), QIcon::Selected, QIcon::Off);
        MainWidget->setWindowIcon(icon);
        horizontalLayout = new QHBoxLayout(MainWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        m_main_label = new QLabel(MainWidget);
        m_main_label->setObjectName(QString::fromUtf8("m_main_label"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_main_label->sizePolicy().hasHeightForWidth());
        m_main_label->setSizePolicy(sizePolicy);
        m_main_label->setMinimumSize(QSize(400, 300));
        m_main_label->setMaximumSize(QSize(2000, 16777215));
        m_main_label->setSizeIncrement(QSize(4, 3));
        m_main_label->setFrameShape(QFrame::Panel);
        m_main_label->setFrameShadow(QFrame::Sunken);
        m_main_label->setLineWidth(2);
        m_main_label->setMidLineWidth(0);

        verticalLayout_2->addWidget(m_main_label);

        m_main_label_2 = new QLabel(MainWidget);
        m_main_label_2->setObjectName(QString::fromUtf8("m_main_label_2"));
        sizePolicy.setHeightForWidth(m_main_label_2->sizePolicy().hasHeightForWidth());
        m_main_label_2->setSizePolicy(sizePolicy);
        m_main_label_2->setMinimumSize(QSize(400, 300));
        m_main_label_2->setMaximumSize(QSize(2000, 16777215));
        m_main_label_2->setSizeIncrement(QSize(4, 3));
        m_main_label_2->setFrameShape(QFrame::Panel);
        m_main_label_2->setFrameShadow(QFrame::Sunken);
        m_main_label_2->setLineWidth(2);
        m_main_label_2->setMidLineWidth(0);

        verticalLayout_2->addWidget(m_main_label_2);


        horizontalLayout->addLayout(verticalLayout_2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        m_button_start = new QPushButton(MainWidget);
        m_button_start->setObjectName(QString::fromUtf8("m_button_start"));

        verticalLayout->addWidget(m_button_start);

        m_button_open = new QPushButton(MainWidget);
        m_button_open->setObjectName(QString::fromUtf8("m_button_open"));

        verticalLayout->addWidget(m_button_open);


        horizontalLayout->addLayout(verticalLayout);


        retranslateUi(MainWidget);

        QMetaObject::connectSlotsByName(MainWidget);
    } // setupUi

    void retranslateUi(QWidget *MainWidget)
    {
        MainWidget->setWindowTitle(QApplication::translate("MainWidget", "Form", 0, QApplication::UnicodeUTF8));
        m_main_label->setText(QApplication::translate("MainWidget", "Hello World!", 0, QApplication::UnicodeUTF8));
        m_main_label_2->setText(QApplication::translate("MainWidget", "Hello World!", 0, QApplication::UnicodeUTF8));
        m_button_start->setText(QApplication::translate("MainWidget", "Start", 0, QApplication::UnicodeUTF8));
        m_button_open->setText(QApplication::translate("MainWidget", "Open", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class MainWidget: public Ui_MainWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAIN_WIDGET_H

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "afterprocessslot.h"
#include "imgprocess.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
// signals:
//    void finish(const QString & msg);//自定义信号

private slots:
    void on_pushButton_clicked();
private:
    AfterProcessSlot afterProcessSlot;
    ImgProcessCore processcore;

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H

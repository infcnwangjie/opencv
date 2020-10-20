#ifndef AFTERPROCESS_H
#define AFTERPROCESS_H

#include <QMessageBox>



class AfterProcessSlot:public QObject{

    Q_OBJECT
   public:
       AfterProcessSlot() {}
    public slots:
       void receiveMsg(const QString & msg)
       {
           QMessageBox::information(NULL,"after process",msg);
       }
};


#endif // AFTERPROCESS_H

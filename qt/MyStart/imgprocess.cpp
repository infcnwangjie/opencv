#include "imgprocess.h"

void RoiTemplate::setName(String name){
    this->name=name;
}

string RoiTemplate::getName(){
    return this->name;
}

void RoiTemplate::setRate(double rate){
    this->rate=rate;
}

double RoiTemplate::getRate(){
    return this->rate;
}

void RoiTemplate::setPosition(int x, int y){
    this->x=x;
    this->y=y;
}


int * RoiTemplate::getPosition(){
    int position[]={this->x,this->y};
    return position;
}


void RoiTemplate::setFind(bool hasfind){
   this->find=hasfind;
}

bool RoiTemplate::getFind(){
    return  this->find;
}

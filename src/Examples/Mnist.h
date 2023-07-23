#pragma once
#include "../Matrix.h"
#include "../Network.h"

Matrix* LabelToMatrix(int label);

int MatrixToLabel(const Matrix* matrix);

Matrix*** GetDataset(std::string path,int dataLength, bool format2D = false);

Matrix*** GetFashionDataset(std::string data,std::string label,int& dataLength, bool format2D = false);

void Mnist1();
void Mnist2();

void FashionMnist1();
void FashionMnist2();

double TestAccuracy(Network* network, Matrix*** data, int dataLength);

void LoadAndTest(std::string filename, bool is2D = false);

int ReverseInt(int number);
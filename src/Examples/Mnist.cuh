#pragma once
#include "../Matrix.cuh"
#include "../Network.cuh"

int MatrixToLabel(const Matrix* matrix);

#if USE_GPU
Matrix_GPU*** GetDataset(std::string path,int dataLength, bool format2D = false);

Matrix_GPU* LabelToMatrix(int label);

double TestAccuracy(Network* network, Matrix_GPU*** data, int dataLength);
#else
Matrix*** GetDataset(std::string path,int dataLength, bool format2D = false);

Matrix* LabelToMatrix(int label);

double TestAccuracy(Network* network, Matrix*** data, int dataLength);
#endif

Matrix*** GetFashionDataset(std::string data,std::string label,int& dataLength, bool format2D = false);

void Mnist1();
void Mnist2();

void FashionMnist1();
void FashionMnist2();

void LoadAndTest(std::string filename, bool is2D = false);

int ReverseInt(int number);
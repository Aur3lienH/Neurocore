#pragma once
#include "../Matrix.h"
#include "../Network.h"

Matrix* LabelToMatrix(int label);

int MatrixToLabel(const Matrix* matrix);

Matrix*** GetDataset(std::string path,int dataLength);

void Mnist1();
void Mnist2();

double TestAccuracy(Network* network, Matrix** inputs, Matrix** outputs, int dataLength);

void LoadAndTest(std::string filename);

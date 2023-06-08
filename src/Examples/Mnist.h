#pragma once
#include "../Matrix.h"
#include "../Network.h"

Matrix* LabelToMatrix(int label);

int MatrixToLabel(const Matrix* matrix);

Matrix*** GetDataset(std::string path,int dataLength);

void Mnist();

double TestAccuracy(Network* network, Matrix*** dataset, int dataLength);



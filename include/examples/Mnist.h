#pragma once

#include "matrix/Matrix.cuh"
#include "network/Network.h"

int MatrixToLabel(const MAT* matrix);

MAT*** GetDataset(const std::string& path, int dataLength, bool format2D = false);

MAT* LabelToMatrix(int label);

double TestAccuracy(Network* network, MAT*** data, int dataLength);

MAT*** GetFashionDataset(const std::string& data, const std::string& label, int& dataLength, bool format2D = false);

void Mnist1();

void Mnist2();

void FashionMnist1();

void FashionMnist2();

void LoadAndTest(std::string filename, bool is2D = false);

int ReverseInt(int number);
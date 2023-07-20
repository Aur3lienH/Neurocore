#pragma once
#include "../Matrix.h"
#include "../Network.h"
#include "Mnist.h"

void QuickDraw1();

std::pair<int, int> GetDataLengthAndNumCategories(const std::string& path);

Matrix* LabelToMatrix(int label, int numLabels);

Matrix*** GetQuickdrawDataset(const std::string& path, int dataLength, int numCategories);
#pragma once
#include "../Matrix.h"
#include "../Network.h"
#include "Mnist.h"

void QuickDraw1(int numDrawingsPerCategory = 20000);
void QuickDraw2(int numDrawingsPerCategory = 20000);

std::pair<int, int> GetDataLengthAndNumCategories(const std::string& path, int numDrawingsPerCategory);

Matrix* LabelToMatrix(int label, int numLabels);

Matrix***
GetQuickdrawDataset(const std::string& path, int dataLength, int numCategories, int numDrawingsPerCategory, bool format2D);
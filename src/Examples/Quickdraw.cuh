#pragma once

#include "../Matrix.cuh"
#include "../Network.cuh"
#include "Mnist.cuh"

void QuickDraw1(int numDrawingsPerCategory = 20000);

void QuickDraw2(int numDrawingsPerCategory = 20000);

std::pair<int, int> GetDataLengthAndNumCategories(const std::string& path, int numDrawingsPerCategory);

MAT* LabelToMatrix(int label, int numLabels);

MAT*** GetQuickdrawDataset(const std::string& path, int dataLength, int numCategories, int numDrawingsPerCategory,
                           bool format2D);
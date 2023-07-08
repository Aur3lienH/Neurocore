#pragma once
#include "Matrix.h"
#include <random>
#include <algorithm>
#include <iostream>


class DataLoader
{
public:
    DataLoader(Matrix*** data, int batchSize, int dataLength);
    void Shuffle();
    int dataLength;
    Matrix*** data;

private:
    int batchSize;
    std::random_device rd;
    std::mt19937 rng;
};
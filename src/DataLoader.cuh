#pragma once

#include "Matrix.cuh"
#include <random>
#include <algorithm>
#include <iostream>


class DataLoader
{
public:
#if USE_GPU
    DataLoader(Matrix_GPU*** data, int dataLength);
#else
    DataLoader(Matrix*** data, int dataLength);
#endif

    void Shuffle();

    int dataLength;
#if USE_GPU
    Matrix_GPU*** data;
#else
    Matrix*** data;
#endif
private:

    std::random_device rd;
    std::mt19937 rng;
};
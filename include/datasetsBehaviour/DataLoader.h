#pragma once

#include "matrix/Matrix.cuh"
#include <random>
#include <algorithm>
#include <iostream>


class DataLoader
{
public:
    DataLoader(MAT*** data, int dataLength);

    void Shuffle();

    int dataLength;
    MAT*** data;
private:

    std::random_device rd;
    std::mt19937 rng;
};
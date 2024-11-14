#pragma once

#include <random>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"

namespace py = pybind11;

class DataLoader
{
public:
    DataLoader(MAT*** data, int dataLength);
    DataLoader(py::array_t<float> input, py::array_t<float> output);
    void Shuffle();
    size_t GetSize();
    MAT*** data;
private:
    LayerShape dataFormat;
    size_t dataLength = 0;
    std::random_device rd;
    std::mt19937 rng;
};

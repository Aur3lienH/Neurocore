#pragma once

#include <random>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"

namespace py = pybind11;

template<int rows_in, int cols_in, int dims_in, int rows_out, int cols_out, int dims_out>
class DataLoader
{
public:
    DataLoader(MAT<rows_in,cols_in,dims_in>** input_data, MAT<rows_out,cols_out,dims_out>** output_data, int dataLength);
    DataLoader(py::array_t<float> input, py::array_t<float> output);
    void Shuffle();
    size_t GetSize();
    MAT<rows_in,cols_in,dims_in>** input_data;
    MAT<rows_out,cols_out,dims_out>** output_data;
private:
    LayerShape<rows_in,cols_in,dims_in> dataFormat;
    size_t dataLength = 0;
    std::random_device rd;
    std::mt19937 rng;
};



DataLoader::DataLoader(MAT*** _data, int _dataLength)
{
    data = _data;
    dataLength = _dataLength;
    rng = std::mt19937(rd());
}


void DataLoader::Shuffle()
{
    std::shuffle(data, data + dataLength, rng);
}

size_t DataLoader::GetSize()
{
    return dataLength;
}


#pragma once

#include <random>
#include <algorithm>
#include <iostream>

#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"


template<typename InputShape, typename OutputShape>
class DataLoader
{
public:
    DataLoader(LMAT<InputShape>** input_data, LMAT<OutputShape>** output_data, int dataLength)
    {
        this->input_data = input_data;
        this->output_data = output_data;
        this->dataLength = dataLength;
        rng = std::mt19937(rd());
    }
    //DataLoader(py::array_t<float> input, py::array_t<float> output);
    void Shuffle()
    {
        //Shuffle the same way the input and output
        std::shuffle(input_data, input_data + dataLength, rng);
        std::shuffle(output_data, output_data + dataLength, rng);

    }
    size_t GetSize()
    {
        return dataLength;
    }
    LMAT<InputShape>** input_data;
    LMAT<OutputShape>** output_data;

private:
    size_t dataLength = 0;
    std::random_device rd;
    std::mt19937 rng;
};



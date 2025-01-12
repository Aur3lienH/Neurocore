#pragma once

#include <random>
#include <algorithm>
#include <iostream>

#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"


template<typename Network>
class DataLoader
{
public:
    typedef std::tuple<LMAT<typename Network::InputShape>,LMAT<typename Network::OutputShape>> TrainingPair;
    DataLoader(TrainingPair* data, size_t dataLength)
    {
        this->data = data;
        this->dataLength = dataLength;
        rng = std::mt19937(rd());
    }

    //DataLoader(py::array_t<float> input, py::array_t<float> output);
    void Shuffle()
    {
        std::shuffle(data, data + dataLength, rng);
    }
    size_t GetSize() const
    {
        return dataLength;
    }
    TrainingPair* data;

private:
    size_t dataLength = 0;
    std::random_device rd;
    std::mt19937 rng;
};



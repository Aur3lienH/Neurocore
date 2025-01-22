#pragma once

#include <random>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <pybind11/pybind11.h>
#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"


template<typename Network>
class DataLoader
{
public:
    typedef LMAT<typename Network::InputShape> InputShape;
    typedef LMAT<typename Network::OutputShape> OutputShape;
    InputShape* inputs;
    OutputShape* outputs;

    DataLoader(InputShape* inputs, OutputShape* outputs ,size_t dataLength)
    {
        this->inputs = inputs;
        this->outputs = outputs;
        this->dataLength = dataLength;
        rng = std::mt19937(rd());
    }

    DataLoader(const py::object& inputs_capsule, const py::object& outputs_capsule, size_t dataLength)
    {
        InputShape* inputs = static_cast<InputShape*>(inputs_capsule.cast<py::capsule>().get_pointer());
        OutputShape* outputs = static_cast<OutputShape*>(outputs_capsule.cast<py::capsule>().get_pointer());
        this->inputs = inputs;
        this->outputs = outputs;
        this->dataLength = dataLength;
        rng = std::mt19937(rd());

    }



    void Shuffle()
    {
        for (size_t i = dataLength - 1; i > 0; --i) 
        {
            size_t j = rng() % (i + 1);

            std::swap(inputs[i], inputs[j]);

            std::swap(outputs[i], outputs[j]);
        }
    }

    size_t GetSize() const
    {
        return dataLength;
    }

private:
    size_t dataLength = 0;
    std::random_device rd;
    std::mt19937 rng;
};



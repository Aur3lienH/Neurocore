#include "Matrix.cuh"
#include "iostream"

#pragma once

class LayerShape
{
public:
    //Constructor for 1D neurons layer
    explicit LayerShape(int neuronsCount);

    ~LayerShape();

    //Constructor for 3D neurons layer (ConvLayer,Pooling, ect ...)
    LayerShape(int rows, int cols, int dims);

    LayerShape(int rows, int cols, int dims, int size);


    //Convert the format of the layer to an array of matrix.
    [[nodiscard]] Matrix* ToMatrix() const;

    //The size of each dimensions
    int* dimensions;

    //The number of dimensions
    int size;

    static LayerShape* Load(std::ifstream& reader);

    void Save(std::ofstream& save);

    [[nodiscard]] std::string GetDimensions() const;
};
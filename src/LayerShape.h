#include "Matrix.h"
#pragma once

class LayerShape
{
public:
    //Constructor for 1D neurons layer
    LayerShape(int neuronsCount);

    //Constructor for 3D neurons layer (ConvLayer,Pooling, ect ...)
    LayerShape(int rows, int cols, int size);

    //Convert the format of the layer to an array of matrix.
    Matrix* ToMatrix();

    //The size of each dimensions
    int* dimensions;

    //The number of dimensions
    int size;
};
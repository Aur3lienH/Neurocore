#pragma once

class LayerShape
{
public:
    LayerShape(int neuronsCount);
    LayerShape(int rows, int cols, int size);
    int* dimensions;
    int size;
};
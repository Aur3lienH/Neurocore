#include "LayerShape.h"


LayerShape::LayerShape(int neuronsCount)
{
    dimensions = new int[3] {neuronsCount,1,1};
    size = 1;
}

LayerShape::LayerShape(int rows, int cols, int _size)
{
    dimensions = new int[3] {rows, cols,_size};
    size = 3;
}
#include "LayerShape.h"


LayerShape::LayerShape(int neuronsCount)
{
    dimensions = new int[1] {neuronsCount};
    size = 1;
}

LayerShape::LayerShape(int rows, int cols, int _size)
{
    dimensions = new int[3] {rows, cols,_size};
    size = 3;
}
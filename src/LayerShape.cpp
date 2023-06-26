#include "LayerShape.h"
#include "Matrix.h"


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

//Convert the format of the layer to an array of matrix.
Matrix* LayerShape::ToMatrix()
{
    Matrix* res = new Matrix[dimensions[2]];
    for (int i = 0; i < dimensions[2]; i++)
    {
        res[i] = Matrix(dimensions[0],dimensions[1]);
    }

    return res;
    
}
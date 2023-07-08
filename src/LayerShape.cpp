#include "LayerShape.h"
#include "Matrix.h"
#include "Layer.h"
#include "iostream"

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

LayerShape::LayerShape(int rows, int cols, int dims, int size)
{
    dimensions = new int[3] {rows, cols,dims};
    this->size = size;
}

//Convert the format of the layer to an array of matrix.
Matrix* LayerShape::ToMatrix()
{
    if(dimensions[2] == 1)
    {
        return new Matrix(dimensions[0],dimensions[1]);
    }
    Matrix* res = new Matrix[dimensions[2]];
    for (int i = 0; i < dimensions[2]; i++)
    {
        res[i] = Matrix(dimensions[0],dimensions[1]);
    }

    return res;
    
}


LayerShape* LayerShape::Load(std::ifstream& reader)
{
    int rows;
    int cols;
    int dims;
    int size;
    reader.read(reinterpret_cast<char*>(&rows),sizeof(int));
    reader.read(reinterpret_cast<char*>(&cols),sizeof(int));
    reader.read(reinterpret_cast<char*>(&dims),sizeof(int));
    reader.read(reinterpret_cast<char*>(&size),sizeof(int));
    return new LayerShape(rows,cols,dims,size);
}

void LayerShape::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(dimensions),sizeof(int));
    writer.write(reinterpret_cast<char*>(dimensions+1),sizeof(int));
    writer.write(reinterpret_cast<char*>(dimensions+2),sizeof(int));
    writer.write(reinterpret_cast<char*>(&size),sizeof(int));
}


std::string LayerShape::GetDimensions()
{
    return  "(" + std::to_string(dimensions[0]) + "," + std::to_string(dimensions[1]) + "," +  std::to_string(dimensions[2]) + ")";
}
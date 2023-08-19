#include "LayerShape.cuh"
#include "Matrix.cuh"
#include "Layer.cuh"
#include "iostream"

LayerShape::LayerShape(const int neuronsCount)
{
    dimensions = new int[3]{neuronsCount, 1, 1};
    size = 1;
}

LayerShape::LayerShape(const int rows, const int cols, const int _size)
{
    dimensions = new int[3]{rows, cols, _size};
    size = 3;
}

LayerShape::LayerShape(const int rows, const int cols, const int dims, const int size)
{
    dimensions = new int[3]{rows, cols, dims};
    this->size = size;
}

//Convert the format of the layer to an array of matrix.
MAT* LayerShape::ToMatrix() const
{
    if (dimensions[2] == 1)
    {
        return new MAT(dimensions[0], dimensions[1]);
    }
    auto* res = new MAT[dimensions[2]];
    for (int i = 0; i < dimensions[2]; i++)
    {
        res[i] = MAT(dimensions[0], dimensions[1]);
    }

    return res;

}


LayerShape* LayerShape::Load(std::ifstream& reader)
{
    int rows;
    int cols;
    int dims;
    int size;
    reader.read(reinterpret_cast<char*>(&rows), sizeof(int));
    reader.read(reinterpret_cast<char*>(&cols), sizeof(int));
    reader.read(reinterpret_cast<char*>(&dims), sizeof(int));
    reader.read(reinterpret_cast<char*>(&size), sizeof(int));
    return new LayerShape(rows, cols, dims, size);
}

void LayerShape::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(dimensions), sizeof(int));
    writer.write(reinterpret_cast<char*>(dimensions + 1), sizeof(int));
    writer.write(reinterpret_cast<char*>(dimensions + 2), sizeof(int));
    writer.write(reinterpret_cast<char*>(&size), sizeof(int));
}


std::string LayerShape::GetDimensions() const
{
    return "(" + std::to_string(dimensions[0]) + "," + std::to_string(dimensions[1]) + "," +
           std::to_string(dimensions[2]) + ")";
}

LayerShape::~LayerShape()
{
    delete[] dimensions;
}

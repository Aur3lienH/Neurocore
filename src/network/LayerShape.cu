#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"
#include <iostream>

//Convert the format of the layer to an array of matrix.
template<int rows, int cols, int dims, int size>
MAT* LayerShape<rows, cols, dims, size>::ToMatrix() const
{
    if (dims == 1)
    {
        return new MAT(rows, cols);
    }
    auto* res = new MAT[dims];
    for (int i = 0; i < dims; i++)
    {
        res[i] = MAT(rows, cols);
    }

    return res;

}


template<int rows, int cols, int dims, int size>
LayerShape<rows, cols, dims, size>* LayerShape<rows, cols, dims, size>::Load(std::ifstream& reader)
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

template<int rows, int cols, int dims, int size>
void LayerShape<rows, cols, dims, size>::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(rows), sizeof(int));
    writer.write(reinterpret_cast<char*>(cols), sizeof(int));
    writer.write(reinterpret_cast<char*>(dims), sizeof(int));
    writer.write(reinterpret_cast<char*>(size), sizeof(int));
}

template<int rows, int cols, int dims, int size>
std::string LayerShape<rows, cols, dims, size>::GetDimensions() const
{
    return "(" + std::to_string(rows) + "," + std::to_string(cols) + "," +
           std::to_string(dims) + ")";
}
#pragma once

#include "matrix/Matrix.cuh"
#include "tools/Vector.h"




template<int x = 1, int y = 1, int z = 1, int size = 1>
class LayerShape
{
public:
    //Convert the format of the layer to an array of matrix.
    [[nodiscard]] constexpr MAT<x,y,z>* ToMatrix() const;

//    constexpr static LayerShape* Load(std::ifstream& reader);

//    constexpr void Save(std::ofstream& save);

    [[nodiscard]] constexpr std::string GetDimensions() const;
};





//Convert the format of the layer to an array of matrix.
template<int rows, int cols, int dims, int size>
constexpr MAT<rows,cols,dims>* LayerShape<rows, cols, dims, size>::ToMatrix() const
{
    if (dims == 1)
    {
        return new MAT(rows, cols);
    }
    auto* res = new MAT<rows,cols>[dims];
    for (int i = 0; i < dims; i++)
    {
        res[i] = MAT();
    }

    return res;
}

template<int rows, int cols, int dims, int size>
constexpr std::string LayerShape<rows, cols, dims, size>::GetDimensions() const
{
    return "(" + std::to_string(rows) + "," + std::to_string(cols) + "," +
           std::to_string(dims) + ")";
}


/*

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

*/

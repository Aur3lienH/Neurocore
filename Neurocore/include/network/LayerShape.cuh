#pragma once

#include "matrix/Matrix.cuh"
#include "tools/Vector.h"




template<int _x = 1, int _y = 1, int _z = 1, int _size = 1>
class LayerShape
{
public:
    static constexpr int x = _x;
    static constexpr int y = _y;
    static constexpr int z = _z;
    //Convert the format of the layer to an array of matrix.
    [[nodiscard]] MAT<x,y,z>* ToMatrix() const {
        if (z == 1)
        {
            return new MAT<x,y,1>();
        }
        auto* res = new MAT<x,y,z>();
        for (int i = 0; i < z; i++)
        {
            res[i] = MAT<x,y,z>();
        }

        return res;
    }

//    constexpr static LayerShape* Load(std::ifstream& reader);

//    constexpr void Save(std::ofstream& save);

    [[nodiscard]] constexpr static std::string GetDimensions();
};




template<int rows, int cols, int dims, int size>
constexpr std::string LayerShape<rows, cols, dims, size>::GetDimensions()
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

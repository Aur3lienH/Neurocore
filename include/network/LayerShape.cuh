#pragma once

#include "matrix/Matrix.cuh"
#include "tools/Vector.h"


template<int x = 1, int y = 1, int z = 1, int size = 1>
class LayerShape
{
public:
    //Convert the format of the layer to an array of matrix.
    [[nodiscard]] constexpr MAT<x,y,z>* ToMatrix() const;

    constexpr static LayerShape* Load(std::ifstream& reader);

    constexpr void Save(std::ofstream& save);

    [[nodiscard]] constexpr std::string GetDimensions() const;
};


//Convert the format of the layer to an array of matrix.
template<int rows, int cols, int dims, int size>
MAT<rows,cols,dims>* LayerShape<rows, cols, dims, size>::ToMatrix() const
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
std::string LayerShape<rows, cols, dims, size>::GetDimensions() const
{
    return "(" + std::to_string(rows) + "," + std::to_string(cols) + "," +
           std::to_string(dims) + ")";
}

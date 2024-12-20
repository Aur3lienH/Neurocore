#include "matrix/Matrix.cuh"
#include "iostream"

#pragma once

template<int rows, int cols, int dims, int size>
class LayerShape
{
public:
    //Convert the format of the layer to an array of matrix.
    constexpr [[nodiscard]] MAT* ToMatrix() const;

    constexpr static LayerShape* Load(std::ifstream& reader);

    constexpr void Save(std::ofstream& save);

    constexpr [[nodiscard]] std::string GetDimensions() const;
};

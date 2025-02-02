#pragma once
#include "matrix/Matrix.cuh"

enum class Opti
{
    Constant,
    Adam
};

template<typename Derived>
class Optimizer final
{
public:
    virtual ~Optimizer() = default;
    static void Compile(int size)
    {
        Derived::Compile(size);
    }

    template<int rows, int cols, int dims>
    static void Compute(MAT<rows,cols,dims>* gradient, MAT<rows,cols,dims>* parameters, int offset = 0)
    {
        Derived::Compute(gradient, parameters, offset);
    }
};
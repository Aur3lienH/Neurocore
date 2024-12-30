#pragma once
#include "matrix/Matrix.cuh"

template<typename Derived>
class Optimizer {
public:
    virtual ~Optimizer() = default;
    static void Compile(int size)
    {
        Derived::Compile(size);
    }
    static void Compute(MAT* gradient, MAT* parameters, int offset = 0)
    {
        Derived::Compute(gradient, parameters, offset);
    }
};
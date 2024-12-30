#pragma once
#include "matrix/Matrix.cuh"

template<typename Derived>
class Loss {
public:
    template<int rows, int cols, int dims>
    static double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target) {
        return Derived::ComputeCost(output, target);
    }

    template<int rows, int cols, int dims>
    static void CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result) {
        Derived::ComputeCostDerivative(output, target, result);
    }
};
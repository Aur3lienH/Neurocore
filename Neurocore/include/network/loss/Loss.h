#pragma once
#include "matrix/Matrix.cuh"
#include "network/loss/MSE.cuh"

template<typename Derived>
class Loss final
{
public:
    static double Cost(const MAT<Derived::Rows,Derived::Cols,Derived::Dims>* output, const MAT<Derived::Rows,Derived::Cols,Derived::Dims>* target) {
        return Derived::Cost(output, target);
    }

    static void CostDerivative(const MAT<Derived::Rows,Derived::Cols,Derived::Dims>* output, const MAT<Derived::Rows,Derived::Cols,Derived::Dims>* target, MAT<Derived::Rows,Derived::Cols,Derived::Dims>* result) {
        Derived::CostDerivative(output, target, result);
    }
};
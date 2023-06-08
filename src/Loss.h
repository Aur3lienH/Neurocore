#pragma once
#include "Matrix.h"


class Loss
{
public:
    Loss();
    virtual double Cost(const Matrix* output, const Matrix* target) = 0;
    virtual void CostDerivative(const Matrix* output, const Matrix* target, Matrix* result) = 0;
};

class MSE : public Loss
{
public:
    double Cost(const Matrix* output, const Matrix* target);
    void CostDerivative(const Matrix* output, const Matrix* target, Matrix* result);
};

class CrossEntropy : public Loss
{
public:
    double Cost(const Matrix* output, const Matrix* target);
    void CostDerivative(const Matrix* output,const Matrix* target, Matrix* result);
};

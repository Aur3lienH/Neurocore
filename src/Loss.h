#pragma once

#include "Matrix.h"


class Loss
{
public:
    Loss();

    virtual ~Loss() = default;

    virtual double Cost(const Matrix* output, const Matrix* target) = 0;

    virtual void CostDerivative(const Matrix* output, const Matrix* target, Matrix* result) = 0;

    static Loss* Read(std::ifstream& reader);

    void Save(std::ofstream& writer);

protected:
    int ID;
};

class MSE : public Loss
{
public:
    MSE();

    double Cost(const Matrix* output, const Matrix* target) override;

    void CostDerivative(const Matrix* output, const Matrix* target, Matrix* result) override;
};

class CrossEntropy : public Loss
{
public:
    CrossEntropy();

    double Cost(const Matrix* output, const Matrix* target) override;

    void CostDerivative(const Matrix* output, const Matrix* target, Matrix* result) override;
};

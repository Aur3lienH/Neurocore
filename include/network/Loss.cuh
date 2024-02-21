#pragma once

#include "matrix/Matrix.cuh"


class Loss
{
public:
    Loss();

    virtual ~Loss() = default;

    virtual double Cost(const MAT* output, const MAT* target) = 0;

    virtual void CostDerivative(const MAT* output, const MAT* target, MAT* result) = 0;

    static Loss* Read(std::ifstream& reader);

    void Save(std::ofstream& writer);

protected:
    int ID;
};

class MSE : public Loss
{
public:
    MSE();

    double Cost(const MAT* output, const MAT* target) override;

    void CostDerivative(const MAT* output, const MAT* target, MAT* result) override;
};

class CrossEntropy : public Loss
{
public:
    CrossEntropy();

    double Cost(const MAT* output, const MAT* target) override;

    void CostDerivative(const MAT* output, const MAT* target, MAT* result) override;
};

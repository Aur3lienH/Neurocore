#pragma once
#include "Matrix.h"

enum class Opti
{
    Constant,
    Adam
};


class Optimizer
{
public:
    virtual void Compile(int size) = 0;
    virtual void Compute(Matrix* gradient, Matrix* parameters, int offset = 0) = 0;
};


class Constant : public Optimizer
{
public:
    Constant(double learningRate = 0.01);
    void Compile(int size) override;
    void Compute(Matrix* gradient, Matrix* parameters, int offset) override;

private:
    double learningRate;
};


class Adam : public Optimizer
{
public:
    Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double gamma = 10e-8);
    void Compile(int size) override;
    void Compute(Matrix* gradient, Matrix* parameters, int offset) override;
private:
    double alpha;
    volatile const double beta1;
    volatile const double beta2;

    double adjBeta1;
    double adjBeta2;


    double gamma;
    double* momentum1;
    double* momentum2;

    double* biasCorrectedMomentum1;
    double* biasCorrectedMomentum2;

};
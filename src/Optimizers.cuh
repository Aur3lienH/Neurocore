#pragma once

#include "Matrix.cuh"

enum class Opti
{
    Constant,
    Adam
};


class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual void Compile(int size) = 0;

    virtual void Compute(MAT* gradient, MAT* parameters, int offset = 0) = 0;
};


class Constant : public Optimizer
{
public:
    explicit Constant(double learningRate = 0.01);

    void Compile(int size) override;

    void Compute(MAT* gradient, MAT* parameters, int offset) override;

private:
    double learningRate;
};


class Adam : public Optimizer
{
public:
    explicit Adam(double alpha = 0.00025, double beta1 = 0.9, double beta2 = 0.999, double gamma = 10e-8);

    void Compile(int size) override;

    void Compute(MAT* gradient, MAT* parameters, int offset) override;

private:
    double alpha;
    volatile const double beta1;
    volatile const double beta2;

    double adjBeta1 = 1.0;
    double adjBeta2 = 1.0;


    double gamma;
    double* momentum1 = nullptr;
    double* momentum2 = nullptr;

    double* biasCorrectedMomentum1 = nullptr;
    double* biasCorrectedMomentum2 = nullptr;

};
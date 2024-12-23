#pragma once

#include "matrix/Matrix.cuh"

enum class Opti
{
    Constant,
    Adam
};

template <typename OptiImpl>
class Optimizer
{
public:
    virtual ~Optimizer() = default;

    void Compile(int size);

    void Compute(MAT* gradient, MAT* parameters, int offset = 0);
};

template <typename OptiImpl>
void Optimizer<OptiImpl>::Compile(int size) {
    static_cast<OptiImpl*>(this)->Compile(size);
}

template <typename OptiImpl>
void Optimizer<OptiImpl>::Compute(MAT* gradient, MAT* parameters, int offset) {
    static_cast<OptiImpl*>(this)->Compute(gradient, parameters, offset);
}


class Constant
{
public:
    explicit Constant(double learningRate = 0.01);

    void Compile(int size);

    void Compute(MAT* gradient, MAT* parameters, int offset);

private:
    double learningRate;
};


class Adam
{
public:
    explicit Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double gamma = 10e-7);

    ~Adam();

    void Compile(int size);

    void Compute(MAT* gradient, MAT* parameters, int offset);

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

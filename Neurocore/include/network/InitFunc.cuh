#pragma once

#include "matrix/Matrix.cuh"
#include <random>
#include <iostream>


//Class in which there are functions to init weights
class WeightsInit final
{
public:
    template<int x = 1, int y = 1, int z = 1>
    static void XavierInit(int inputSize, MAT<x,y,z>* weights);

    template<int x = 1, int y = 1, int z = 1>
    static void NormalizedXavierInit(int inputSize, int outputSize, MAT<x,y,z>* weights);

    template<int x = 1, int y = 1, int z = 1>
    static void HeUniform(int inputSize, MAT<x,y,z>* weights);

private:
    inline static std::mt19937 rng{std::random_device{}()};
};


#include "network/InitFunc.cuh"
#include <cmath>


template<int x, int y, int z>
void WeightsInit::XavierInit(const int inputSize, MAT<x,y,z>* weights)
{
    float upper = 1.0 / sqrt((float) inputSize);
    float lower = -upper;
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#else
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};

template<int x, int y, int z>
void WeightsInit::NormalizedXavierInit(const int inputSize, const int outputSize, MAT<x,y,z>* weights)
{
    float upper = (sqrt(6.0) / sqrt((float) inputSize + (float) outputSize));
    float lower = -upper;
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#else
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};

template<int x, int y, int z>
void WeightsInit::HeUniform(const int inputSize, MAT<x,y,z>* weights)
{
    double limit = std::sqrt(6.0 / inputSize);
    std::uniform_real_distribution<double> distribution(-limit, limit);
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = distribution(rng);
#else
        (*weights)[i] = distribution(rng);
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};


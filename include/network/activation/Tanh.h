#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include "network/activation/Activation.cuh"



class Tanh
{
public:
    Tanh();

#if not USE_GPU

    double Function(double input);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);


    void FeedForward(const MAT* input, MAT* output)
    {
        DefaultFeedForward(input, output, Function);
    }

    void Derivative(const MAT* input, MAT* output)
    {
        DefaultDerivative(input, output, Derive);
    }
};


Tanh::Tanh()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

double Tanh::Function(const double input)
{
    return tanh(input);
}

#endif

double Tanh::Derive(const double input)
{
    return 1 - tanh(input) * tanh(input);
}

MAT* Tanh::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}
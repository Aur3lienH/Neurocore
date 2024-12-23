#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include "network/activation/Activation.cuh"

class SigmoidPrime
{
public:
    SigmoidPrime();

#if not USE_GPU

    static double Function(double input);

#endif

    static double Derive(double input);

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



SigmoidPrime::SigmoidPrime()
{
#if USE_GPU
    throw std::runtime_error("The sigmoid prime class has no meaning on GPU, please use the sigmoid class instead");
#endif
}

#if not USE_GPU

double SigmoidPrime::Function(double input)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

#endif

double SigmoidPrime::Derive(const double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}

MAT* SigmoidPrime::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}



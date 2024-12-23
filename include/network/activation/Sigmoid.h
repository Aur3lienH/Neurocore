#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include "network/activation/Activation.cuh"

class Sigmoid
{
public:
    Sigmoid() {
#if USE_GPU
        checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
        checkCUDNN(
                cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
    }


    static double Function(double input) {
        return 1 / (1 + exp(-input));
    }

    static double Derive(double input)
    {
        return exp(-input) / pow(1 + exp(-input), 2);
    }

    void FeedForward(const MAT* input, MAT* output)
    {
        DefaultFeedForward(input, output, Function);
    }

    void Derivative(const MAT* input, MAT* output)
    {
        DefaultDerivative(input, output, Derive);
    }

    static MAT* InitWeights(int previousNeuronsCount, int NeuronsCount)
    {
#if USE_GPU
        auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
        auto* weights = new MAT(NeuronsCount, previousNeuronsCount, 1, true);
#endif
        WeightsInit::XavierInit(previousNeuronsCount, weights);
        return weights;
    }
};
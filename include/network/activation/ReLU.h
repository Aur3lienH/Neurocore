#pragma once
#include <cmath>
#include <matrix/Matrix.cuh>
#include <network/InitFunc.cuh>
#include <emmintrin.h>

class ReLU
{
public:
    ReLU();

#if not USE_GPU

    void Derivative(const MAT* input, MAT* output);

    double Function(double input);

    void FeedForward(const MAT* input, MAT* output);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);

    MAT* InitBiases(int outputSize);


};



ReLU::ReLU()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(
            cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

void ReLU::FeedForward(const MAT* input, MAT* output)
{
    __m128 zero = _mm_setzero_ps();

    size_t i;
    for (i = 0; i <= input->GetSize() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 result = _mm_max_ps(zero, vals);
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->GetSize(); ++i)
    {
        if ((*input)[i] < 0) (*output)[i] = 0;
    }
}

#endif

#if not USE_GPU

void ReLU::Derivative(const MAT* input, MAT* output)
{
    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0);

    int i;
    for (i = 0; i <= input->GetSize() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 mask = _mm_cmpgt_ps(vals,
                                   zero); // Create a mask where each element is either 0xFFFFFFFFFFFFFFFF if vals > 0 or 0x0 otherwise
        __m128 result = _mm_and_ps(one, mask);  // Set to 1.0 where mask is true
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->GetSize(); ++i)
    {
        (*output)[i] = ((*input)[i] > 0) ? 1.0 : 0.0;
    }
}

double ReLU::Function(const double input)
{
    if (input > 0)
    {
        return input;
    }
    else
    {
        return 0;
    }
}

#endif

double ReLU::Derive(const double input)
{
    if (input > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

MAT* ReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::HeUniform(previousNeuronsCount, weights);
    return weights;
}

MAT* ReLU::InitBiases(const int outputSize)
{
#if USE_GPU
    float* biases = new float[outputSize];
    for (int i = 0; i < outputSize; i++)
        biases[i] = 0.01f;

    Matrix_GPU* res = new Matrix_GPU(outputSize, 1);
    checkCUDA(cudaMemcpy(res->GetData(), biases, outputSize * sizeof(float), cudaMemcpyHostToDevice));
    delete[] biases;

    return res;
#else
    return new MAT(outputSize, 1, 0.01f);
#endif
}
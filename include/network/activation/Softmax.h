#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"



class Softmax
{
public:
    Softmax();

#if USE_GPU

    void FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                     const cudnnTensorDescriptor_t& outputDesc) override;

    void Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                    const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z, const cudnnTensorDescriptor_t& zDesc,
                    MAT* output, const cudnnTensorDescriptor_t& outputDesc) override;

#else

    void FeedForward(const MAT* input, MAT* output);

    void Derivative(const MAT* input, MAT* output);

    double inline Function(double input)
    { return 0; };

#endif

    double inline Derive(double input)
    { return 0; };

    MAT* InitWeights(int inputSize, int outputSize);

};

Softmax::Softmax()
{
}

#if USE_GPU

void Softmax::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                          const cudnnTensorDescriptor_t& outputDesc)
#else

void Softmax::FeedForward(const MAT* input, MAT* output)
#endif
{
#if USE_GPU
    checkCUDNN(cudnnSoftmaxForward(Matrix_GPU::cuda->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &Matrix_GPU::cuda->one, inputDesc, input->GetData(),
                                   &Matrix_GPU::cuda->zero, outputDesc, output->GetData()));
#else
    double sum = 0;
    double max = input[0][0];
    for (int i = 0; i < input->GetSize(); i++)
    {
        if (input[0][i] > max)
        {
            max = input[0][i];
        }
    }

    for (int i = 0; i < input->GetSize(); i++)
    {
        sum += exp(input[0][i] - max);
    }
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = exp(input[0][i] - max) / sum;
    }
#endif
}

MAT* Softmax::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    MAT* weights = new MAT(NeuronsCount, previousNeuronsCount);
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}

#if USE_GPU

void Softmax::Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                         const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z,
                         const cudnnTensorDescriptor_t& zDesc,
                         MAT* output, const cudnnTensorDescriptor_t& outputDesc)
{
    /*checkCUDNN(cudnnSoftmaxBackward(Matrix_GPU::cuda->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &Matrix_GPU::cuda->one, *input->GetDescriptor_1D(), input->GetData(),
                                    *lastDelta->GetDescriptor_1D(), lastDelta->GetData(), &Matrix_GPU::cuda->zero,
                                    *output->GetDescriptor_1D(), output->GetData()));*/

    // The CPU version sets all values of output to one, but as the GPU version of Derivative also multiplies output
    // by lastDelta, we can just copy lastDelta to output
    checkCUDA(cudaMemcpy(output->GetData(), lastDelta->GetData(), output->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
}

#else

void Softmax::Derivative(const MAT* input, MAT* output)
{
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = 1;
    }
}


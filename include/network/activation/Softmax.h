#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"


template<int rows,int prev_rows, int cols = 1, int dims = 1>
class Softmax final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    Softmax();

#if USE_GPU

    static void FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                     const cudnnTensorDescriptor_t& outputDesc) override;

    static void Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                    const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z, const cudnnTensorDescriptor_t& zDesc,
                    MAT* output, const cudnnTensorDescriptor_t& outputDesc) override;

#else

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output);

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output);

#endif

    static MAT<rows,prev_rows,dims>* InitWeights();

    static std::string getName()
    {
        return "Softmax";
    }

};
template<int rows,int prev_rows, int cols, int dims>
Softmax<rows,prev_rows,cols,dims>::Softmax()
{
}

#if USE_GPU

void Softmax::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                          const cudnnTensorDescriptor_t& outputDesc)
#else
template<int rows,int prev_rows, int cols, int dims>
void Softmax<rows,prev_rows,cols,dims>::FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
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

template<int rows,int prev_rows, int cols, int dims>
MAT<rows,prev_rows,dims>* Softmax<rows,prev_rows,cols,dims>::InitWeights()
{
    MAT<rows,prev_rows,dims>* weights = new MAT<rows,prev_rows,dims>();
    WeightsInit::XavierInit(prev_rows, weights);
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

template<int rows,int prev_rows, int cols, int dims>
void Softmax<rows,prev_rows,cols,dims>::Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
{
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = 1;
    }
}

#endif
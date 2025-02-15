#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"


template<int rows,int prev_rows, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class Softmax final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    Softmax();

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output, const Matrix<rows,cols,dims>* lastDelta, const Matrix<rows,cols,dims>* z);

    static void FeedForward(const MAT<rows, cols, dims>* input, MAT<rows, cols, dims>* output);

    static MAT<rows,prev_rows,dims>* InitWeights();

    static std::string getName()
    {
        return "Softmax";
    }

};
template<int rows,int prev_rows, int cols, int dims, bool GPU>
Softmax<rows,prev_rows,cols,dims, GPU>::Softmax()
{
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
void Softmax<rows,prev_rows,cols,dims, GPU>::FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)

{
    if constexpr (GPU)
    {
        cudnnSoftmaxForward(cuda->cudnnHandle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &cuda->one, input->desc, input->GetData(), &cuda->zero, output->desc, output->GetData());
        return;
    }
    else
    {
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
    }
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
MAT<rows,prev_rows,dims>* Softmax<rows,prev_rows,cols,dims, GPU>::InitWeights()
{
    MAT<rows,prev_rows,dims>* weights = new MAT<rows,prev_rows,dims>();
    WeightsInit::XavierInit(prev_rows, weights);
    return weights;
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
void Softmax<rows,prev_rows,cols,dims, GPU>::Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output, [[maybe_unused]] const Matrix<rows,cols,dims>* lastDelta, [[maybe_unused]] const Matrix<rows,cols,dims>* z)
{
    if constexpr (GPU)
    {
        /*checkCUDNN(cudnnSoftmaxBackward(Matrix_GPU::cuda->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                        &Matrix_GPU::cuda->one, *input->GetDescriptor_1D(), input->GetData(),
                                        *lastDelta->GetDescriptor_1D(), lastDelta->GetData(), &Matrix_GPU::cuda->zero,
                                        *output->GetDescriptor_1D(), output->GetData()));*/

        // The CPU version sets all values of output to one, but as the GPU version of Derivative also multiplies output
        // by lastDelta, we can just copy lastDelta to output
        checkCUDA(cudaMemcpy(output->GetData(), lastDelta->GetData(), output->GetSize() * sizeof(float),
                             cudaMemcpyHostToDevice));
        return;
    }
    else
    {
        for (int i = 0; i < input->GetSize(); i++)
        {
            output[0][i] = 1;
        }
    }
}
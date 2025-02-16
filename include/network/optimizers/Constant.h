#pragma once
#include "gpuComputation/CUDA.cuh"

__global__
void ConstantComputeKernel(const float* gradient, float* parameters, const int size, const double learningRate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        parameters[i] -= gradient[i] * learningRate;

}


template<double lr, bool GPU = GPU_DEFAULT>
class Constant final
{
public:
    static void Compile(int size) {}

    template<int rows, int cols, int dims>
    static void Compute(MAT<rows,cols,dims>* gradient, MAT<rows,cols,dims>* parameters, int offset = 0) {

        if constexpr (GPU)
        {
            const int numBlocks = (gradient->GetSize() + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock;
            checkKernel((ConstantComputeKernel<<<numBlocks, cuda->threadsPerBlock>>>(gradient->GetData() + offset, parameters->GetData() + offset, gradient->GetSize(), lr)));
        }
        else
        {
            for (int i = 0; i < gradient->GetSize(); i++) {
                (*parameters)[i] -= (*gradient)[i] * lr;
            }
        }
    }
};
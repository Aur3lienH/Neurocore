#pragma once
#include "Optimizer.h"



template<double lr>
class Constant final
{
public:
    static void Compile(int size) {}

    template<int rows, int cols, int dims>
    static void Compute(MAT<rows,cols,dims>* gradient, MAT<rows,cols,dims>* parameters, int offset = 0) {
#if USE_GPU
        const int numBlocks = (gradient->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
        ConstantComputeKernel<<<numBlocks, Matrix_GPU::cuda->threadsPerBlock>>>(gradient->GetData() + offset, parameters->GetData() + offset, gradient->GetSize(), lr);
        checkCUDA(cudaDeviceSynchronize());
#else
        for (int i = 0; i < gradient->GetSize(); i++) {
            (*parameters)[i] -= (*gradient)[i] * lr;
        }
#endif
    }
};
#pragma once
#include "Optimizer.h"



template<double lr>
class Constant {
public:
    static void Compile(int size) {}

    static void Compute(MAT* gradient, MAT* parameters, int offset = 0) {
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
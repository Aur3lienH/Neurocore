#pragma once
#include "Loss.h"
#include "matrix/Matrix.cuh"
#include <cmath>

template <int rows, int cols, int dims, bool GPU = GPU_DEFAULT>
class CrossEntropy final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;

private:
    static constexpr float EPSILON = 1e-15;

public:
    static double Cost(const MAT<rows, cols, dims>* output, const MAT<rows, cols, dims>* target)
    {
        if constexpr (GPU)
        {
            float* res_d;
            checkCUDA(cudaMalloc(&res_d, output->GetSize() * sizeof(float)));

            checkKernel((CrossEntropyKernel<<<CUDA_KERNEL_ARGS(cuda, output->GetSize())>>>
                (output->GetData(), target->GetData(), res_d, output->GetSize(), EPSILON)));

            float* r;
            checkCUDA(cudaMalloc(&r, sizeof(float)));
            checkKernel((SumKernel<<<1, 1>>>(res_d, output->GetSize(), r)));

            float* r_h = new float[1];
            checkCUDA(cudaMemcpy(r_h, r, sizeof(float), cudaMemcpyDeviceToHost));

            float result = -static_cast<double>(*r_h) / output->GetRows();
            delete[] r_h;
            checkCUDA(cudaFree(r));
            checkCUDA(cudaFree(res_d));

            return result;
        }
        else
        {
            double cost = 0;
            for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
            {
                cost += target->data[i] * log(output->data[i] + EPSILON) +
                    (1 - target->data[i]) * log(1 - output->data[i] + EPSILON);
            }
            return -cost / output->GetRows();
        }
    }

    static void CostDerivative(const MAT<rows, cols, dims>* output,
                               const MAT<rows, cols, dims>* target,
                               MAT<rows, cols, dims>* result)
    {
        if constexpr (GPU)
        {
            const int blocksPerGrid =
                (output->GetSize() + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock;
            checkKernel((CostDerivativeKernel<<<blocksPerGrid, cuda->threadsPerBlock>>>
                (output->GetData(), target->GetData(), result->GetData(), output->GetSize())));
        }
        else
        {
            for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
            {
                if (target->data[i] == 1)
                {
                    result->data[i] = -1 + output->data[i];
                }
                else
                {
                    result->data[i] = output->data[i];
                }
            }
        }
    }
};

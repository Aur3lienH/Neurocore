#pragma once
#include "Loss.h"
#include "matrix/Matrix.cuh"
#include <cmath>

template<int rows, int cols, int dims>
class CrossEntropy final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
private:
    static constexpr float EPSILON = 1e-15;
#if USE_GPU
    // Kernel CUDA pour calculer la cross-entropy par élément
    __global__
    static void CrossEntropyKernel(const float* output, const float* target, float* result, const int size) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            result[i] = target[i] * logf(output[i] + EPSILON) +
                       (1.0f - target[i]) * logf(1.0f - output[i] + EPSILON);
        }
    }

    // Kernel CUDA pour la somme
    __global__
    static void SumKernel(float* arr, const int len, float* res) {
        for (int i = 0; i < len; i++) {
            *res += arr[i];
        }
    }

    // Kernel CUDA pour calculer la dérivée
    __global__
    static void CostDerivativeKernel(const float* output, const float* target, float* result, const int size) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            if (target[i] == 1) {
                result[i] = -1 + output[i];
            } else {
                result[i] = output[i];
            }
        }
    }
#endif

public:
    static double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target) {
#if USE_GPU
        float* res_d;
        checkCUDA(cudaMalloc(&res_d, output->GetSize() * sizeof(float)));

        CrossEntropyKernel<<<Matrix_GPU::cuda->threadsPerBlock, Matrix_GPU::cuda->threadsPerBlock>>>
            (output->GetData(), target->GetData(), res_d, output->GetSize());
        checkCUDA(cudaDeviceSynchronize());

        float* r;
        checkCUDA(cudaMalloc(&r, sizeof(float)));
        SumKernel<<<1, 1>>>(res_d, output->GetSize(), r);
        checkCUDA(cudaDeviceSynchronize());

        float* r_h = new float[1];
        checkCUDA(cudaMemcpy(r_h, r, sizeof(float), cudaMemcpyDeviceToHost));

        float result = -static_cast<double>(*r_h);
        delete[] r_h;
        checkCUDA(cudaFree(r));
        checkCUDA(cudaFree(res_d));

        return result;
#else
        double cost = 0;
        for (int i = 0; i < output->GetRows() * output->GetCols(); i++) {
            cost += target[0][i] * log(output[0][i] + EPSILON) +
                    (1 - target[0][i]) * log(1 - output[0][i] + EPSILON);
        }
        return -cost / output->GetRows();
#endif
    }

    static void CostDerivative(const MAT<rows,cols,dims>* output,
                              const MAT<rows,cols,dims>* target,
                              MAT<rows,cols,dims>* result) {
#if USE_GPU
        const int blocksPerGrid =
            (output->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
        CostDerivativeKernel<<<blocksPerGrid, Matrix_GPU::cuda->threadsPerBlock>>>
            (output->GetData(), target->GetData(), result->GetData(), output->GetSize());
        checkCUDA(cudaDeviceSynchronize());
#else
        for (int i = 0; i < output->GetRows() * output->GetCols(); i++) {
            if (target[0][i] == 1) {
                result[0][i] = -1 + output[0][i];
            } else {
                result[0][i] = output[0][i];
            }
        }
#endif
    }
};
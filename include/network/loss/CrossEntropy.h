#pragma once
#include "matrix/Matrix.cuh"
#include <emmintrin.h>

class CrossEntropy
{
public:
    CrossEntropy();

    template<int rows, int cols, int dims>
    double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target);

    template<int rows, int cols, int dims>
    void CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result);
};


/*
__global__ void sum_kernel(float* array, float* sum, int array_size)
{
    extern __shared__ float partial_sums[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sums[tid] = 0;

    while (i < array_size)
    {
        partial_sums[tid] += array[i];
        i += blockDim.x * gridDim.x;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            partial_sums[tid] += partial_sums[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        sum[blockIdx.x] = partial_sums[0];
    }
}
*/
/*
#if USE_GPU
float Sum(const float* arr_d, const int array_size)
{
    const int BLOCK_SIZE = Matrix_GPU::cuda->threadsPerBlock;
    int num_blocks = std::ceil((float) array_size / BLOCK_SIZE);
    float* sum_d;
    checkCUDA(cudaMalloc(&sum_d, num_blocks * sizeof(float)));
    while (num_blocks > 1)
    {
        const int num_threads = num_blocks;
        checkCUDA(cudaMalloc(&sum_d, num_threads * sizeof(float)));

        sum_kernel<<<num_blocks, BLOCK_SIZE>>>(const_cast<float*>(arr_d), sum_d, num_threads);

        num_blocks = std::ceil((float) num_blocks / BLOCK_SIZE);
    }

    float res;
    checkCUDA(cudaMemcpy(&res, sum_d, sizeof(float), cudaMemcpyDeviceToHost));
    return res;
}
#endif

 */
/*
__global__
void CrossEntropyKernel(const float* output, const float* target, float* result, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        result[i] = target[i] * log(output[i] + 1e-15) + (1 - target[i]) * log(1 - output[i] + 1e-15);
}

__global__
void SumK(float* arr, const int len, float* res)
{
    for (int i = 0; i < len; i++)
        *res += arr[i];
}
*/
template<int rows, int cols, int dims>
double CrossEntropy::Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target)
{
#if USE_GPU
    float* res_d;
    checkCUDA(cudaMalloc(&res_d, output->GetSize() * sizeof(float)));
    CrossEntropyKernel<<<Matrix_GPU::cuda->threadsPerBlock, Matrix_GPU::cuda->threadsPerBlock>>>(output->GetData(),
                                                                                                 target->GetData(),
                                                                                                 res_d,
                                                                                                 output->GetSize());

    checkCUDA(cudaDeviceSynchronize());
    float* r;
    checkCUDA(cudaMalloc(&r, sizeof(float)));
    SumK<<<1, 1>>>(res_d, output->GetSize(), r);
    checkCUDA(cudaDeviceSynchronize());
    float* r_h = new float[1];
    checkCUDA(cudaMemcpy(r_h, r, sizeof(float), cudaMemcpyDeviceToHost));
    return -static_cast<double>(*r_h);
    //return -Sum(res_d, output->GetSize() /* /Matrix_GPU::cuda->threadsPerBlock*/) / output->GetRows();

    /*Matrix outputCPU(output->GetRows(), output->GetCols(), output->GetData_CPU_1D());
    Matrix targetCPU(target->GetRows(), target->GetCols(), target->GetData_CPU_1D());
    double cost = 0;

    for (int i = 0; i < output->GetMatrixSize(); i++)
        cost += targetCPU[i] * log(outputCPU[i] + 1e-15) + (1 - targetCPU[i]) * log(1 - outputCPU[i] + 1e-15);

    return -cost / output->GetRows();
#else
    double cost = 0;

    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
        cost += target[0][i] * log(output[0][i] + 1e-15) + (1 - target[0][i]) * log(1 - output[0][i] + 1e-15);

*/
    return -cost / output->GetRows();
#else
    return 0;

#endif
}

/*
__global__
void CostDerivativeKernel(const float* output, const float* target, float* result, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        if (target[i] == 1)
        {
            result[i] = -1 + output[i];
        }
        else
        {
            result[i] = output[i];
        }
    }
}
*/
template<int rows, int cols, int dims>
void CrossEntropy::CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result)
{
#if USE_GPU
    const int blocksPerGrid =
            (output->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
    CostDerivativeKernel<<<blocksPerGrid, Matrix_GPU::cuda->threadsPerBlock>>>(output->GetData(), target->GetData(),
                                                                               result->GetData(), output->GetSize());
    checkCUDA(cudaDeviceSynchronize());
#else
    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
    {
        if (target[0][i] == 1)
        {
            result[0][i] = -1 + output[0][i];
        }
        else
        {
            result[0][i] = output[0][i];
        }
    }
#endif
}

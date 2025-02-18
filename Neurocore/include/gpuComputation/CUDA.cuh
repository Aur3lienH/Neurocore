#ifndef DEEPLEARNING_CUDA_CUH
#define DEEPLEARNING_CUDA_CUH


#include "cudnn.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

#ifdef USE_GPU
constexpr bool GPU_DEFAULT = true;
#else
constexpr bool GPU_DEFAULT = false;
#endif

#define checkCUDNN(expression) \
{ \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
                  << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

static const char* cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
//Macro for checking cuda errors following a cuda launch or api call
#define checkCUDA(expression) {                                          \
 cudaError_t e = (expression);                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


#define checkKernel(kernel_expression) { \
    (kernel_expression); \
    cudaError_t err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Kernel Execution Failed: " << cudaGetErrorString(err) << std::endl; \
        exit(0); \
    } \
}

#define checkCUBLAS(expression)                                                                        \
    {                                                                                                 \
        if (expression != CUBLAS_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    expression, cublasGetErrorEnum(expression), __FILE__, __LINE__);                                 \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

#define CUDA_KERNEL_ARGS(cuda, data_length) (data_length + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock, cuda->threadsPerBlock

class CUDA
{
public:
    CUDA()
    {
        checkCUDNN(cudnnCreate(&cudnnHandle));
        checkCUBLAS(cublasCreate_v2(&cublasHandle));
    }

    ~CUDA()
    {
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUBLAS(cublasDestroy_v2(cublasHandle));
    }

    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
    const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;
    const int threadsPerBlock = 256;
};

inline CUDA* cuda = new CUDA();

__global__ static void initializeArray_kernel(float *array, float value, int n);
__global__ static void scalarMult_kernel(float* arr, float val, float* res, int n);
__global__ static void transpose_kernel(float *A, float *A_T, int rows, int cols);
__global__ static void leakyReluFeedForward(float* input, float *output, int n, float alpha);
__global__ static void leakyReluDerivative(float *input, float* output, int n, float alpha);
__global__ static void CrossEntropyKernel(const float* output, const float* target, float* result, int size, float EPSILON);
__global__ static void CostDerivativeKernel(const float* output, const float* target, float* result, int size);
__global__ static void SumKernel(float* arr, int len, float* res);
__global__ static void MSEDerivativeKernel(const float* output, const float* target, float* result, int size);
__global__ static void ConstantComputeKernel(const float* gradient, float* parameters, int size, double learningRate);

__global__ static void initializeArray_kernel(float* array, float value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        array[idx] = value;
    }
}

__global__ static void scalarMult_kernel(float* arr, float val, float* res, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    res[idx] = arr[idx] * val;
}

__global__ static void transpose_kernel(float* A, float* A_T, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        A_T[x * rows + y] = A[y * cols + x];
    }
}

__global__ static void leakyReluFeedForward(float* input, float* output, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] < 0 ? alpha * input[idx] : input[idx];;
    }
}

__global__ static void leakyReluDerivative(float* input, float* output, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] < 0 ? alpha : 1;;
    }
}

__global__ static void CrossEntropyKernel(const float* output, const float* target, float* result, const int size,
                                   float EPSILON)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        result[i] = target[i] * logf(output[i] + EPSILON) +
            (1.0f - target[i]) * logf(1.0f - output[i] + EPSILON);
    }
}

__global__ static void CostDerivativeKernel(const float* output, const float* target, float* result, const int size)
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

__global__ static void SumKernel(float* arr, const int len, float* res)
{
    for (int i = 0; i < len; i++)
    {
        *res += arr[i];
    }
}

__global__
static void MSEDerivativeKernel(const float* output, const float* target,
                         float* result, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = output[idx] - target[idx];
    }
}

__global__
static void ConstantComputeKernel(const float* gradient, float* parameters, const int size, const double learningRate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        parameters[i] -= gradient[i] * learningRate;

}

#endif
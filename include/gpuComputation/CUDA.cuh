#ifndef DEEPLEARNING_CUDA_CUH
#define DEEPLEARNING_CUDA_CUH


#include "cudnn.h"
#include "cublas_v2.h"
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

__global__ void initializeArray_kernel(float *array, float value, int n);
__global__ void scalarMult_kernel(float* arr, float val, float* res, int n);
__global__ void transpose_kernel(float *A, float *A_T, int rows, int cols);

#endif
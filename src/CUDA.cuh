//
// Created by mat on 19/08/23.
//

#ifndef DEEPLEARNING_CUDA_CUH
#define DEEPLEARNING_CUDA_CUH

#if USE_GPU

#include "cudnn.h"
#include "cublas_v2.h"

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

#define checkCUBLAS(expression)                                                                        \
    {                                                                                                 \
        if (expression != CUBLAS_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    expression, cublasGetErrorEnum(expression), __FILE__, __LINE__);                                 \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

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
    const float one = 1.0f, zero = 0.0f;
    const int threadsPerBlock = 256;
};


#endif
#endif //DEEPLEARNING_CUDA_CUH
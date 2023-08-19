#if USE_GPU
//
// Created by mat on 19/08/23.
//

#ifndef DEEPLEARNING_CUDA_CUH
#define DEEPLEARNING_CUDA_CUH

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
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
                    expression, _cublasGetErrorEnum(expression), __FILE__, __LINE__);                                 \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

class CUDA
{
public:
    CUDA()
    {
        checkCUDNN(cudnnCreate(&cudnnHandle));
        //checkCUDNN(cublasCreate_v2(&cublasHandle));
    }

    ~CUDA()
    {
        checkCUDNN(cudnnDestroy(cudnnHandle));
        //checkCUDNN(cublasDestroy_v2(cublasHandle));
    }

    cudnnHandle_t cudnnHandle;
    const float alpha = 1.0f, beta = 0.0f;
};


#endif //DEEPLEARNING_CUDA_CUH
#endif
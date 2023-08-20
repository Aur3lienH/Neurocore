#include "Examples/Mnist.cuh"

#define TEST_ACTIVATION_BACKWARD 0

int main()
{
#if TEST_ACTIVATION_BACKWARD

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnCreate(&cudnnHandle);
    cublasCreate_v2(&cublasHandle);

    cudnnActivationDescriptor_t activationDesc;
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));

    Matrix values(28, 28);
    for (int i = 0; i < 28 * 28; i++)
        values[i] = i - 5;

    float* forward_input_d;
    checkCUDA(cudaMalloc(&forward_input_d, 28 * 28 * sizeof(float)));
    checkCUDA(cudaMemcpy(forward_input_d, values.GetData(), 28 * 28 * sizeof(float), cudaMemcpyHostToDevice));
    cudnnTensorDescriptor_t forwardInputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&forwardInputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardInputDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          28,
                                          28));

    float* forward_output_d;
    checkCUDA(cudaMalloc(&forward_output_d, 28 * 28 * sizeof(float)));
    cudnnTensorDescriptor_t forwardOutputDescriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          28,
                                          28));

    checkCUDNN(cudnnActivationForward(cudnnHandle,
                                      activationDesc,
                                      &alpha,
                                      forwardInputDescriptor,
                                      forward_input_d,
                                      &beta,
                                      forwardOutputDescriptor,
                                      forward_output_d));

    float* forward_output = new float[28 * 28];
    checkCUDA(cudaMemcpy(forward_output, forward_output_d, 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Values: \n";
    for (int i = 0; i < 28 * 28; i++)
        std::cout << values[i] << " ";

    std::cout << std::endl;

    std::cout << "Forward output: \n";
    for (int i = 0; i < 28 * 28; i++)
        std::cout << forward_output[i] << " ";

    float* ones_d;
    checkCUDA(cudaMalloc(&ones_d, 28 * 28 * sizeof(float)));
    checkCUDA(cudaMemset(ones_d, 1, 28 * 28 * sizeof(float)));
    cudnnTensorDescriptor_t onesDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&onesDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(onesDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          28,
                                          28));

    float* backward_output_d;
    checkCUDA(cudaMalloc(&backward_output_d, 28 * 28 * sizeof(float)));
    cudnnTensorDescriptor_t backward_output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&backward_output_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(backward_output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          28,
                                          28));

    float* backward_delta_output_d;
    checkCUDA(cudaMalloc(&backward_delta_output_d, 28 * 28 * sizeof(float)));
    cudnnTensorDescriptor_t backward_delta_output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&backward_delta_output_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(backward_delta_output_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          28,
                                          28));

    checkCUDNN(cudnnActivationBackward(cudnnHandle,
                                       activationDesc,
                                       &alpha,
                                       forwardOutputDescriptor,
                                       forward_output_d,
                                       onesDesc,
                                       ones_d,
                                       backward_output_desc,
                                       backward_output_d,
                                       &beta, backward_delta_output_desc,
                                       backward_delta_output_d));

    float* backward_output = new float[28 * 28];
    checkCUDA(cudaMemcpy(backward_output, backward_output_d, 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost));

    float* backward_delta_output = new float[28 * 28];
    checkCUDA(cudaMemcpy(backward_delta_output, backward_delta_output_d, 28 * 28 * sizeof(float),
                         cudaMemcpyDeviceToHost));

    std::cout << std::endl;

    std::cout << "Backward output: \n";
    for (int i = 0; i < 28 * 28; i++)
        std::cout << backward_output[i] << " ";

    std::cout << std::endl;

    std::cout << "Backward delta output: \n";
    for (int i = 0; i < 28 * 28; i++)
        std::cout << backward_delta_output[i] << " ";

    return 0;
#endif
    Mnist1();
    //QuickDraw2(10000);

#if USE_GPU
    delete Matrix_GPU::cuda;
#endif
    return 0;
    //LoadAndTest("./Models/MNIST_11.net",true);
}
#include "Examples/Mnist.cuh"
#include "Quickdraw.cuh"

int main()
{
    //Mnist2();
    QuickDraw2(10000);

#if USE_GPU
    delete Matrix_GPU::cuda;
#endif
    return 0;
    //LoadAndTest("./Models/MNIST_11.net",true);
}


/*#include "cudnn.h"
#include <iostream>
#include "Matrix.cuh"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#include <cudnn.h>
#include <iostream>

using namespace std;


int main()
{
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;

    cudnnConvolutionFwdAlgo_t falgo;
    cudnnConvolutionBwdFilterAlgo_t b_falgo;
    cudnnConvolutionBwdDataAlgo_t b_dalgo;

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_filter = nullptr;
    float* d_bias = nullptr;

    int input_n = 1;
    int input_c = 1;
    int input_h = 5;
    int input_w = 5;

    // output size
    int output_n = input_n;
    int output_c = 2;
    int output_h = 1;
    int output_w = 1;

    // kernel size
    int filter_h = 3;
    int filter_w = 3;

    // alpha, beta
    float one = 1.f;
    float zero = 0.f;

    std::cout << "[" << __LINE__ << "]" << std::endl;

    cudnnCreate(&cudnn);

    std::cout << "[" << __LINE__ << "]" << std::endl;

    // Create Resources
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnCreateTensorDescriptor(&bias_desc);

    std::cout << "[" << __LINE__ << "]" << std::endl;

    // Initilziae resources
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, filter_h, filter_w);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, 0,
                                    1, 1,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &output_n, &output_c, &output_h,
                                          &output_w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h,
                               output_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1);

    int weight_size = output_c * input_c * filter_h * filter_w;
    int bias_size = output_c;

    std::cout << "input  size: " << input_n << " " << input_c << " " << input_h << " " << input_w << std::endl;
    std::cout << "output size: " << output_n << " " << output_c << " " << output_h << " " << output_w << std::endl;

    std::cout << "[" << __LINE__ << "]" << std::endl;

    // convolution
    size_t workspace_size = 0;
    size_t temp_size = 0;
    float* d_workspace = nullptr;
    cudnnConvolutionFwdAlgoPerfStruct* perf_results = new cudnnConvolutionFwdAlgoPerfStruct[1];

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, filter_desc, conv_desc, output_desc, 1, nullptr,
                                           perf_results);
    falgo = perf_results[0].algo;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, falgo, &temp_size);
    workspace_size = max(workspace_size, temp_size);

    // convolution (bwd - filter)
    cudnnConvolutionBwdFilterAlgoPerfStruct* b_fperf_results = new cudnnConvolutionBwdFilterAlgoPerfStruct[1];
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn, input_desc, output_desc, conv_desc, filter_desc,
                                                  1, nullptr, b_fperf_results);
    b_falgo = b_fperf_results[0].algo;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, input_desc, output_desc, conv_desc, filter_desc, b_falgo,
                                                   &temp_size);
    workspace_size = max(workspace_size, temp_size);

    // convolution (bwd - data)
    cudnnConvolutionBwdDataAlgoPerfStruct* b_dperf_results = new cudnnConvolutionBwdDataAlgoPerfStruct[1];
    cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn, filter_desc, output_desc, conv_desc, input_desc,
                                                1, nullptr, b_dperf_results);
    b_dalgo = b_dperf_results[0].algo;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_desc, output_desc, conv_desc, input_desc, b_dalgo,
                                                 &temp_size);
    workspace_size = max(workspace_size, temp_size);

    std::cout << "workspace size: " << workspace_size << std::endl;
    std::cout << "[" << __LINE__ << "]" << std::endl;

    // allocate memory space
    cudaMalloc((void**) &d_input, sizeof(float) * input_n * input_c * input_h * input_w);
    cudaMalloc((void**) &d_filter, sizeof(float) * weight_size);
    cudaMalloc((void**) &d_output, sizeof(float) * output_n * output_c * output_h * output_w);
    cudaMalloc((void**) &d_workspace, sizeof(float) * workspace_size);
    cudaMalloc((void**) &d_bias, sizeof(float) * bias_size);

    cudaMemset(d_bias, 0, sizeof(float) * bias_size);

    float* h_input = new float[input_n * input_c * input_h * input_w];
    float* h_filter = new float[weight_size];

    for (int i = 0; i < input_n * input_c * input_h * input_w; i++)
        h_input[i] = i;
    for (int i = 0; i < weight_size; i++)
        h_filter[i] = 0;
    h_filter[1 * 3 + 1] = 1;
    h_filter[3 * 3 + 1 * 3 + 1] = 2;
    cudaMemcpy(d_input, h_input, sizeof(float) * input_n * input_c * input_h * input_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(float) * weight_size, cudaMemcpyHostToDevice);

    std::cout << "[" << __LINE__ << "]" << std::endl;

    // Forward
    checkCUDNN(cudnnConvolutionForward(cudnn, &one, input_desc, d_input, filter_desc, d_filter, conv_desc, falgo,
                                       d_workspace, workspace_size, &zero, output_desc, d_output));
    checkCUDNN(cudnnAddTensor(cudnn, &one, bias_desc, d_bias, &one, output_desc, d_output));
    (cudaGetLastError());

    std::cout << "[" << __LINE__ << "]" << std::endl;

    Matrix input(input_h, input_w, input_c, h_input);
    for (int d = 0; d < input_c; d++)
    {
        std::cout << input << std::endl;
        input.GoToNextMatrix();
    }
    input.ResetOffset();
    float* f_output_h = new float[output_n * output_c * output_h * output_w];
    cudaMemcpy(f_output_h, d_output, sizeof(float) * output_n * output_c * output_h * output_w,
               cudaMemcpyDeviceToHost);
    Matrix output(output_h, output_w, output_c, f_output_h);
    for (int d = 0; d < output_c; d++)
    {
        std::cout << output << std::endl;
        output.GoToNextMatrix();
    }
    output.ResetOffset();

    // backward
    checkCUDNN(cudnnConvolutionBackwardBias(cudnn, &one, output_desc, d_output, &zero, bias_desc, d_bias));
    checkCUDNN(
            cudnnConvolutionBackwardFilter(cudnn, &one, input_desc, d_input, output_desc, d_output, conv_desc, b_falgo,
                                           d_workspace, workspace_size, &zero, filter_desc, d_filter));
    checkCUDNN(
            cudnnConvolutionBackwardData(cudnn, &one, filter_desc, d_filter, output_desc, d_output, conv_desc, b_dalgo,
                                         d_workspace, workspace_size, &zero, input_desc, d_input));
    (cudaGetLastError());

    std::cout << "[" << __LINE__ << "]" << std::endl;

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(bias_desc);

    std::cout << "[" << __LINE__ << "]" << std::endl;

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaFree(d_bias);

    cudnnDestroy(cudnn);

    std::cout << "[" << __LINE__ << "]" << std::endl;
}*/

/*int main(void)
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    Matrix image(5, 5);
    for (int d = 0; d < image.GetDims(); d++)
        for (int i = 0; i < image.GetRows(); i++)
            for (int j = 0; j < image.GetCols(); j++)
                image[d * image.GetRows() * image.GetCols() + i * image.GetCols() + j] = i * image.GetCols() + j;

    const int output_h = image.GetRows() - 2, output_w = image.GetCols() - 2;
    const int numFilters = 2;

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,

                                          CUDNN_TENSOR_NCHW,

                                          CUDNN_DATA_FLOAT,

                                          1,

                                          image.GetDims(),

                                          image.GetRows(),

                                          image.GetCols()));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,

                                          CUDNN_TENSOR_NCHW,

                                          CUDNN_DATA_FLOAT,

                                          1,

                                          numFilters,

                                          output_h,

                                          output_w));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,

                                          CUDNN_DATA_FLOAT,

                                          CUDNN_TENSOR_NCHW,

                                          numFilters,

                                          image.GetDims(),
                                          3,
                                          3));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,

                                               0,

                                               0,

                                               1,

                                               1,

                                               1,

                                               1,

                                               CUDNN_CROSS_CORRELATION,

                                               CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    const int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerfStruct* convolution_algorithms = new cudnnConvolutionFwdAlgoPerfStruct[requested_algo_count];
    checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   requested_algo_count,
                                                   &returned_algo_count,
                                                   convolution_algorithms));
    convolution_algorithm = convolution_algorithms[0].algo;

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;

    int o_batch_size{0}, o_channels{0}, o_height{0}, o_width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &o_batch_size,
                                                     &o_channels,
                                                     &o_height,
                                                     &o_width));
    if (numFilters != o_channels)
    {
        std::cerr << "Input/output o_channels do not match network configuration."
                  << std::endl;
        return 1;
    }
    if (output_h != o_height || output_w != o_width)
    {
        std::cerr << "Input/output dimensions do not match network configuration."
                  << std::endl;
        return 1;
    }
    if (o_batch_size != 1)
    {
        std::cerr << "Invalid batch size." << std::endl;
        return 1;
    }

    void* d_workspace{nullptr};
    if (workspace_bytes)
        cudaMalloc(&d_workspace, workspace_bytes);

    const int image_bytes = o_batch_size * image.GetDims() * image.GetRows() * image.GetCols() * sizeof(float);
    const int output_bytes = o_batch_size * numFilters * output_h * output_w * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.GetData(), image_bytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);


    const int kernelSize = numFilters * image.GetDims() * 3 * 3;
    float* h_filter = new float[kernelSize];
    for (int i = 0; i < kernelSize; i++)
        h_filter[i] = 0;
    h_filter[1 * 3 + 1] = 1;
    h_filter[3 * 3 + 1 * 3 + 1] = 2;


    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, kernelSize * sizeof(float));
    cudaMemcpy(d_kernel, h_filter, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));

    float* h_output = new float[output_bytes / sizeof(float)];
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < image.GetDims(); i++)
    {
        std::cout << image << std::endl;
        image.GoToNextMatrix();
    }
    image.ResetOffset();
    CloneMatrix res(output_h, output_w, numFilters, h_output);
    for (int i = 0; i < numFilters; i++)
    {
        std::cout << res << std::endl;
        res.GoToNextMatrix();
    }
    res.ResetOffset();

    //delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

    return 0;;
}*/

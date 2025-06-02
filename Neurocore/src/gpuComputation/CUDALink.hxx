#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"



// Forward declarations of CUDA kernels
__global__ void initializeArray_kernel(float *array, float value, int n);
__global__ void scalarMult_kernel(float* arr, float val, float* res, int n);
__global__ void transpose_kernel(float *A, float *A_T, int rows, int cols);
__global__ void leakyReluFeedForward(float* input, float *output, int n, float alpha);
__global__ void leakyReluDerivative(float *input, float* output, int n, float alpha);
__global__ void CrossEntropyKernel(const float* output, const float* target, float* result, int size, float EPSILON);
__global__ void CostDerivativeKernel(const float* output, const float* target, float* result, int size);
__global__ void SumKernel(float* arr, int len, float* res);
__global__ void MSEDerivativeKernel(const float* output, const float* target, float* result, int size);
__global__ void ConstantComputeKernel(const float* gradient, float* parameters, int size, double learningRate);

// Wrapper function implementations
void initializeArray_kernel_link(float *array, float value, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    initializeArray_kernel<<<grid_size, block_size>>>(array, value, n);
    cudaDeviceSynchronize();
}

void scalarMult_kernel_link(float* arr, float val, float* res, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    scalarMult_kernel<<<grid_size, block_size>>>(arr, val, res, n);
    cudaDeviceSynchronize();
}

void transpose_kernel_link(float *A, float *A_T, int rows, int cols) {
    dim3 block_size(16, 16);  // 2D block for matrix operations
    dim3 grid_size((cols + block_size.x - 1) / block_size.x, 
                   (rows + block_size.y - 1) / block_size.y);
    transpose_kernel<<<grid_size, block_size>>>(A, A_T, rows, cols);
    cudaDeviceSynchronize();
}

void leakyReluFeedForward_link(float* input, float *output, int n, float alpha) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    leakyReluFeedForward<<<grid_size, block_size>>>(input, output, n, alpha);
    cudaDeviceSynchronize();
}

void leakyReluDerivative_link(float *input, float* output, int n, float alpha) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    leakyReluDerivative<<<grid_size, block_size>>>(input, output, n, alpha);
    cudaDeviceSynchronize();
}

void CrossEntropy_link(const float* output, const float* target, float* result, int size, float EPSILON) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    CrossEntropyKernel<<<grid_size, block_size>>>(output, target, result, size, EPSILON);
    cudaDeviceSynchronize();
}

void CostDerivative_link(const float* output, const float* target, float* result, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    CostDerivativeKernel<<<grid_size, block_size>>>(output, target, result, size);
    cudaDeviceSynchronize();
}

void Sum_link(float* arr, int len, float* res) {
    int block_size = 256;
    int grid_size = (len + block_size - 1) / block_size;
    SumKernel<<<grid_size, block_size>>>(arr, len, res);
    cudaDeviceSynchronize();
}

void MSEDerivative_link(const float* output, const float* target, float* result, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    MSEDerivativeKernel<<<grid_size, block_size>>>(output, target, result, size);
    cudaDeviceSynchronize();
}

void ConstantCompute_link(const float* gradient, float* parameters, int size, double learningRate) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    ConstantComputeKernel<<<grid_size, block_size>>>(gradient, parameters, size, learningRate);
    cudaDeviceSynchronize();
}
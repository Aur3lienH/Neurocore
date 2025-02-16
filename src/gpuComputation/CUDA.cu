#include "gpuComputation/CUDA.cuh"

__global__ void initializeArray_kernel(float* array, float value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        array[idx] = value;
    }
}

__global__ void scalarMult_kernel(float* arr, float val, float* res, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    res[idx] = arr[idx] * val;
}

__global__ void transpose_kernel(float* A, float* A_T, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows)
    {
        A_T[x * rows + y] = A[y * cols + x];
    }
}

__global__ void leakyReluFeedForward(float* input, float* output, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] < 0 ? alpha * input[idx] : input[idx];;
    }
}

__global__ void leakyReluDerivative(float* input, float* output, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] < 0 ? alpha : 1;;
    }
}

__global__ void CrossEntropyKernel(const float* output, const float* target, float* result, const int size,
                                   float EPSILON)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        result[i] = target[i] * logf(output[i] + EPSILON) +
            (1.0f - target[i]) * logf(1.0f - output[i] + EPSILON);
    }
}

__global__ void CostDerivativeKernel(const float* output, const float* target, float* result, const int size)
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

__global__ void SumKernel(float* arr, const int len, float* res)
{
    for (int i = 0; i < len; i++)
    {
        *res += arr[i];
    }
}

__global__
void MSEDerivativeKernel(const float* output, const float* target,
                         float* result, const int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = output[idx] - target[idx];
    }
}

__global__
void ConstantComputeKernel(const float* gradient, float* parameters, const int size, const double learningRate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        parameters[i] -= gradient[i] * learningRate;

}
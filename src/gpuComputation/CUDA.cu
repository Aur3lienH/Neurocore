#include "gpuComputation/CUDA.cuh"

__global__ void initializeArray_kernel(float *array, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
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

__global__ void transpose_kernel(float *A, float *A_T, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        A_T[x * rows + y] = A[y * cols + x];
    }
}
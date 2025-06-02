#pragma once
#include "gpuComputation/CUDA.cuh"



void initializeArray_kernel_link(float *array, float value, int n);
void scalarMult_kernel_link(float* arr, float val, float* res, int n);
void transpose_kernel_link(float *A, float *A_T, int rows, int cols);
void leakyReluFeedForward_link(float* input, float *output, int n, float alpha);
void leakyReluDerivative_link(float *input, float* output, int n, float alpha);
void CrossEntropy_link(const float* output, const float* target, float* result, int size, float EPSILON);
void CostDerivative_link(const float* output, const float* target, float* result, int size);
void Sum_link(float* arr, int len, float* res);
void MSEDerivative_link(const float* output, const float* target, float* result, int size);
void ConstantCompute_link(const float* gradient, float* parameters, int size, double learningRate);



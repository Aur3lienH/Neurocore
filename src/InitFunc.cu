#include "InitFunc.cuh"
#include <cmath>

void XavierInit(const int inputSize, MAT* weights)
{
    float upper = 1.0 / sqrt((float) inputSize);
    float lower = -upper;
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#else
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};


void NormalizedXavierInit(const int inputSize, const int outputSize, MAT* weights)
{
    float upper = (sqrt(6.0) / sqrt((float) inputSize + (float) outputSize));
    float lower = -upper;
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#else
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};

void HeInit(const int inputSize, MAT* weights)
{
    float range = sqrt(2.0 / (float) inputSize);
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = (rand() / ((float) RAND_MAX) - 0.5) * 2 * range;
#else
        weights[0][i] = (rand() / ((float) RAND_MAX) - 0.5) * 2 * range;
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};


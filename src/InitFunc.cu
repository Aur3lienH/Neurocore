#include "InitFunc.cuh"
#include <cmath>

std::mt19937 WeightsInit::rng = std::mt19937(std::random_device{}());


void WeightsInit::XavierInit(const int inputSize, MAT* weights)
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


void WeightsInit::NormalizedXavierInit(const int inputSize, const int outputSize, MAT* weights)
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

void WeightsInit::HeUniform(const int inputSize, MAT* weights)
{
    double limit = std::sqrt(6.0 / inputSize);
    std::uniform_real_distribution<double> distribution(-limit, limit);
#if USE_GPU
    Matrix m(weights->GetRows(), weights->GetCols(), weights->GetDims());
#endif

    for (int i = 0; i < weights->GetSize(); i++)
    {
#if USE_GPU
        m[i] = distribution(rng);
#else
        (*weights)[i] = distribution(rng);
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(weights->GetData(), m.GetData(), weights->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
#endif
};


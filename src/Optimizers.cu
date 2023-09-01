#include "Optimizers.cuh"
#include "cmath"


Constant::Constant(const double learningRate)
{
    this->learningRate = learningRate;
}

void Constant::Compile(const int size)
{

}


__global__
void ConstantComputeKernel(const float* gradient, float* parameters, const int size, const double learningRate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        parameters[i] -= gradient[i] * learningRate;

}

void Constant::Compute(MAT* gradient, MAT* parameters, const int offset)
{
#if USE_GPU
    const int numBlocks =
            (gradient->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
    ConstantComputeKernel<<<numBlocks, Matrix_GPU::cuda->threadsPerBlock>>>(gradient->GetData() + offset,
                                                                            parameters->GetData() + offset,
                                                                            gradient->GetSize(), learningRate);
    checkCUDA(cudaDeviceSynchronize());
#else

    for (int i = 0; i < gradient->GetSize(); i++)
    {
        (*parameters)[i] -= (*gradient)[i] * learningRate;
    }
#endif
}


Adam::Adam(const double alpha, const double _beta1, const double _beta2, const double gamma) : beta1(_beta1),
                                                                                               beta2(_beta2)
{
    this->alpha = alpha;
    this->gamma = gamma;
    adjBeta1 = beta1;
    adjBeta2 = beta2;
}

void Adam::Compile(const int size)
{
    if (momentum1 == nullptr)
    {
#if USE_GPU
        checkCUDA(cudaMalloc(&momentum1, size * sizeof(double)));
#else
        momentum1 = new double[size];
        for (int i = 0; i < size; i++)
        {
            momentum1[i] = 0;
        }
#endif
    }
    if (momentum2 == nullptr)
    {
#if USE_GPU
        checkCUDA(cudaMalloc(&momentum2, size * sizeof(double)));
#else
        momentum2 = new double[size];
        for (int i = 0; i < size; i++)
        {
            momentum2[i] = 0;
        }
#endif
    }

    if (biasCorrectedMomentum1 == nullptr)
    {
#if USE_GPU
        checkCUDA(cudaMalloc(&biasCorrectedMomentum1, size * sizeof(double)));
#else
        biasCorrectedMomentum1 = new double[size];
        for (int i = 0; i < size; i++)
        {
            biasCorrectedMomentum1[i] = 0;
        }
#endif
    }

    if (biasCorrectedMomentum2 == nullptr)
    {
#if USE_GPU
        checkCUDA(cudaMalloc(&biasCorrectedMomentum2, size * sizeof(double)));
#else
        biasCorrectedMomentum2 = new double[size];
        for (int i = 0; i < size; i++)
        {
            biasCorrectedMomentum2[i] = 0;
        }
#endif
    }

}

#if USE_GPU

__global__
void
AdamComputeKernel(float* gradient, float* parameters, double* momentum1, double* momentum2,
                  double* biasCorrectedMomentum1, double* biasCorrectedMomentum2, const int size, const double alpha,
                  const double gamma, const double beta1, const double beta2, const double adjBeta1,
                  const double adjBeta2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        const double g = gradient[idx];

        momentum1[idx] = beta1 * momentum1[idx] + (1 - beta1) * g;
        momentum2[idx] = beta2 * momentum2[idx] + (1 - beta2) * g * g;

        biasCorrectedMomentum1[idx] = momentum1[idx] / (1 - adjBeta1);
        biasCorrectedMomentum2[idx] = momentum2[idx] / (1 - adjBeta2);

        parameters[idx] =
                parameters[idx] - alpha * biasCorrectedMomentum1[idx] / (sqrt(biasCorrectedMomentum2[idx]) + gamma);
    }
}

#endif

void Adam::Compute(MAT* _gradient, MAT* parameters, const int offset)
{
    double* _momentum1 = momentum1 + offset;
    double* _momentum2 = momentum2 + offset;

    double* _biasCorrectedMomentum1 = biasCorrectedMomentum1 + offset;
    double* _biasCorrectedMomentum2 = biasCorrectedMomentum2 + offset;

#if USE_GPU
    const int numBlocks =
            (_gradient->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
    AdamComputeKernel<<<numBlocks, Matrix_GPU::cuda->threadsPerBlock>>>(_gradient->GetData() + offset,
                                                                        parameters->GetData() + offset,
                                                                        _momentum1, _momentum2,
                                                                        _biasCorrectedMomentum1,
                                                                        _biasCorrectedMomentum2,
                                                                        _gradient->GetSize(),
                                                                        alpha, gamma, beta1, beta2, adjBeta1,
                                                                        adjBeta2);

#else
    for (int i = 0; i < _gradient->GetSize(); i++)
    {
        const double gradient = (*_gradient)[i];

        _momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * gradient;
        _momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * gradient * gradient;

        _biasCorrectedMomentum1[i] = _momentum1[i] / (1 - adjBeta1);
        _biasCorrectedMomentum2[i] = _momentum2[i] / (1 - adjBeta2);

        (*parameters)[i] =
                (*parameters)[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
    }
#endif


    adjBeta1 *= beta1;
    adjBeta2 *= beta2;
}

Adam::~Adam()
{
#if USE_GPU
    checkCUDA(cudaFree(momentum1));
    checkCUDA(cudaFree(momentum2));
    checkCUDA(cudaFree(biasCorrectedMomentum1));
    checkCUDA(cudaFree(biasCorrectedMomentum2));
#else
    delete[] momentum1;
    delete[] momentum2;
    delete[] biasCorrectedMomentum1;
    delete[] biasCorrectedMomentum2;
#endif
}



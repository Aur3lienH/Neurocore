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
        momentum1 = new double[size];
        for (int i = 0; i < size; i++)
        {
            momentum1[i] = 0;
        }
    }
    if (momentum2 == nullptr)
    {
        momentum2 = new double[size];
        for (int i = 0; i < size; i++)
        {
            momentum2[i] = 0;
        }
    }

    if (biasCorrectedMomentum1 == nullptr)
    {
        biasCorrectedMomentum1 = new double[size];
        for (int i = 0; i < size; i++)
        {
            biasCorrectedMomentum1[i] = 0;
        }
    }

    if (biasCorrectedMomentum2 == nullptr)
    {
        biasCorrectedMomentum2 = new double[size];
        for (int i = 0; i < size; i++)
        {
            biasCorrectedMomentum2[i] = 0;
        }
    }

}

void Adam::Compute(MAT* _gradient, MAT* parameters, const int offset)
{
    double* _momentum1 = momentum1 + offset;
    double* _momentum2 = momentum2 + offset;

    double* _biasCorrectedMomentum1 = biasCorrectedMomentum1 + offset;
    double* _biasCorrectedMomentum2 = biasCorrectedMomentum2 + offset;

#if USE_GPU
    Matrix gradientCPU(_gradient->GetRows(), _gradient->GetCols(), _gradient->GetData_CPU());
    Matrix parametersCPU(parameters->GetRows(), parameters->GetCols(), parameters->GetData_CPU());
#endif


    for (int i = 0; i < _gradient->GetSize(); i++)
    {
#if USE_GPU
        const double gradient = gradientCPU[i];
#else
        const double gradient = (*_gradient)[i];
#endif

        _momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * gradient;
        _momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * gradient * gradient;

        _biasCorrectedMomentum1[i] = _momentum1[i] / (1 - adjBeta1);
        _biasCorrectedMomentum2[i] = _momentum2[i] / (1 - adjBeta2);
#if USE_GPU
        parametersCPU[i] =
                parametersCPU[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
#else
        (*parameters)[i] =
                (*parameters)[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
#endif
    }

    adjBeta1 *= beta1;
    adjBeta2 *= beta2;

#if USE_GPU
    checkCUDA(cudaMemcpy(parameters->GetData(), parametersCPU.GetData(), parametersCPU.GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
#endif
}

Adam::~Adam()
{
    delete[] momentum1;
    delete[] momentum2;
    delete[] biasCorrectedMomentum1;
    delete[] biasCorrectedMomentum2;
}



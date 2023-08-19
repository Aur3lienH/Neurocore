#include "Optimizers.cuh"
#include "Layer.cuh"
#include "cmath"


Constant::Constant(const double learningRate)
{
    this->learningRate = learningRate;
}

void Constant::Compile(const int size)
{

}

void Constant::Compute(MAT* gradient, MAT* parameters, const int offset)
{
#if USE_GPU
    std::cout << "Constant::Compute kernel not implemented for GPU yet" << std::endl;
    Matrix gradientCPU(gradient->GetRows(), gradient->GetCols(), gradient->GetDims());
    Matrix parametersCPU(parameters->GetRows(), parameters->GetCols(), parameters->GetDims());
    checkCUDA(cudaMemcpy(gradientCPU.GetData(), gradient->GetData(), gradient->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(parametersCPU.GetData(), parameters->GetData(), parameters->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
#endif

    for (int i = 0; i < gradient->GetSize(); i++)
    {
#if USE_GPU
        parametersCPU[i] -= gradientCPU[i] * learningRate;
#else
        (*parameters)[i] -= (*gradient)[i] * learningRate;
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(parameters->GetData(), parametersCPU.GetData(), parameters->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
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
    std::cout << "Adam::Compute kernel not implemented for GPU yet" << std::endl;
    Matrix gradientCPU(_gradient->GetRows(), _gradient->GetCols(), _gradient->GetDims());
    Matrix parametersCPU(parameters->GetRows(), parameters->GetCols(), parameters->GetDims());
    checkCUDA(cudaMemcpy(gradientCPU.GetData(), _gradient->GetData(), _gradient->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(parametersCPU.GetData(), parameters->GetData(), parameters->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
#endif

    for (int i = 0; i < _gradient->GetSize(); i++)
    {
#if USE_GPU
        double gradient = gradientCPU[i];
#else
        double gradient = (*_gradient)[i];
#endif

        _momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * gradient;
        _momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * gradient * gradient;

        _biasCorrectedMomentum1[i] = _momentum1[i] / (1 - adjBeta1);
        _biasCorrectedMomentum2[i] = _momentum2[i] / (1 - adjBeta2);

#if USE_GPU
        parametersCPU[i] -= alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
#else
        (*parameters)[i] =
                (*parameters)[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
#endif
    }


    adjBeta1 *= beta1;
    adjBeta2 *= beta2;

#if USE_GPU
    checkCUDA(cudaMemcpy(parameters->GetData(), parametersCPU.GetData(), parameters->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
#endif
}



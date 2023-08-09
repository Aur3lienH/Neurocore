#include "Optimizers.h"
#include "Layer.h"
#include "cmath"


Constant::Constant(const double learningRate)
{
    this->learningRate = learningRate;
}

void Constant::Compile(const int size)
{

}

void Constant::Compute(Matrix* gradient, Matrix* parameters, const int offset)
{
    for (int i = 0; i < gradient->size(); i++)
    {
        (*parameters)[i] -= (*gradient)[i] * learningRate;
    }
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


void Adam::Compute(Matrix* _gradient, Matrix* parameters, const int offset)
{
    double* _momentum1 = momentum1 + offset;
    double* _momentum2 = momentum2 + offset;

    double* _biasCorrectedMomentum1 = biasCorrectedMomentum1 + offset;
    double* _biasCorrectedMomentum2 = biasCorrectedMomentum2 + offset;


    for (int i = 0; i < _gradient->size(); i++)
    {
        double gradient = (*_gradient)[i];

        _momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * gradient;
        _momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * gradient * gradient;

        _biasCorrectedMomentum1[i] = _momentum1[i] / (1 - adjBeta1);
        _biasCorrectedMomentum2[i] = _momentum2[i] / (1 - adjBeta2);

        (*parameters)[i] = (*parameters)[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
    }



    adjBeta1 *= beta1;
    adjBeta2 *= beta2;

}



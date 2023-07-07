#include "Optimizers.h"
#include "Layer.h"
#include "cmath"


Constant::Constant(double learningRate)
{
    this->learningRate = learningRate;
}

void Constant::Compile(int size)
{

}

void Constant::Compute(Matrix* gradient, Matrix* parameters, int offset)
{
    for (int i = 0; i < gradient->size(); i++)
    {
        (*parameters)[i] -= (*gradient)[i] * learningRate;
    }
}




Adam::Adam(double alpha, double _beta1, double _beta2, double gamma) : beta1(_beta1), beta2(_beta2)
{
    this->alpha = alpha;
    this->gamma = gamma;
    adjBeta1 = beta1;
    adjBeta2 = beta2;
}

void Adam::Compile(int size)
{
    momentum1 = new double[size];
    momentum2 = new double[size];

    biasCorrectedMomentum1 = new double[size];
    biasCorrectedMomentum2 = new double[size];
    
}


void Adam::Compute(Matrix* _gradient, Matrix* parameters, int offset)
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
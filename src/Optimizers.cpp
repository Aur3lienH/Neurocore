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

void Constant::Compute(Matrix* gradient, Matrix* parameters)
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


void Adam::Compute(Matrix* _gradient, Matrix* parameters)
{
    for (int i = 0; i < _gradient->size(); i++)
    {
        double gradient = (*_gradient)[i];
        momentum1[i] = beta1 * momentum1[i] + (1 - beta1) * gradient;
        momentum2[i] = beta2 * momentum2[i] + (1 - beta2) * gradient * gradient;

        biasCorrectedMomentum1[i] = momentum1[i] / (1 - adjBeta1);
        biasCorrectedMomentum2[i] = momentum2[i] / (1 - adjBeta2);

        (*parameters)[i] = (*parameters)[i] - alpha * biasCorrectedMomentum1[i] / (sqrt(biasCorrectedMomentum2[i]) + gamma);
    }

    adjBeta1 *= beta1;
    adjBeta1 *= beta2;
    
}
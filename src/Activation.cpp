#include "Activation.h"
#include "InitFunc.h"
#include <math.h>


Activation::Activation()
{
}

std::string Activation::getName() const
{
    return name;
}

Matrix* Activation::InitBiases(int outputSize)
{
    Matrix* Biases = new Matrix(outputSize, 1);
    for (int i = 0; i < outputSize; i++)
    {
        Biases[0][i] = 0;
    }
    return Biases;
}

void Activation::FeedForward(const Matrix* input, Matrix* output)
{
    for (int i = 0; i < input->getCols() * input->getRows(); i++)
    {
        output[0][i] = Function(input[0][i]);
    }
}

void Activation::Derivative(const Matrix * input, Matrix* output)
{
    for (int i = 0; i < input->getCols() * input->getRows(); i++)
    {
        output[0][i] = Derivate(input[0][i]);
    }

}

Sigmoid::Sigmoid()
{
    name = "Sigmoid";
}

double Sigmoid::Function(double input)
{
    return 1 / (1 + exp(-input));
}

double Sigmoid::Derivate(double input)
{
    return exp(-input) / pow(1 + exp(-input), 2);
}

Matrix* Sigmoid::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

SigmoidPrime::SigmoidPrime()
{
    name = "SigmoidPrime";
}

double SigmoidPrime::Function(double input)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

double SigmoidPrime::Derivate(double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}

Matrix* SigmoidPrime::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

ReLU::ReLU()
{
    name = "ReLU";
}

double ReLU::Function(double input)
{
    if(input > 0)
    {
        return input;
    }
    else
    {
        return 0;
    }
}

double ReLU::Derivate(double input)
{
    if(input > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

Matrix* ReLU::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    HeInit(previousNeuronsCount, weights);
    return weights;
}

LeakyReLU::LeakyReLU(double _alpha)
{
    alpha = _alpha;
}

double LeakyReLU::Function(double input)
{
    return input > 0 ? input : 0.01 * input;
}

double LeakyReLU::Derivate(double input)
{
    return input > 0 ? 1 : 0.01;
}

Matrix* LeakyReLU::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    HeInit(previousNeuronsCount, weights);
    return weights;
}

Softmax::Softmax()
{
    name = "Softmax";
}

void Softmax::FeedForward(const Matrix* input, Matrix* output)
{
    double sum = 0;
    double max = input[0][0];
    for (int i = 0; i < input->getCols()*input->getRows(); i++)
    {
        if(input[0][i] > max)
        {
            max = input[0][i];
        }
    }
    
    for (int i = 0; i < input->getCols() * input->getRows(); i++)
    {
        sum += exp(input[0][i] - max);
    }
    for (int i = 0; i < input->getCols() * input->getRows(); i++)
    {
        output[0][i] = exp(input[0][i] - max) / sum;
    }
}

void Softmax::Derivative(const Matrix* input, Matrix* output)
{
    for (int i = 0; i < input->getCols() * input->getRows(); i++)
    {
        output[0][i] = 1;
    }
}

Matrix* Softmax::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

double Softmax::Function(double input)
{
    return 0;
}

double Softmax::Derivate(double input)
{
    return 0;
}

Tanh::Tanh()
{
    name = "Tanh";
}

double Tanh::Function(double input)
{
    return tanh(input);
}

double Tanh::Derivate(double input)
{
    return 1 - tanh(input) * tanh(input);
}

Matrix* Tanh::InitWeights(int previousNeuronsCount, int NeuronsCount)
{
    Matrix* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}








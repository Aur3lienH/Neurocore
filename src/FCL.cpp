#include "FCL.h"
#include <iostream>
#include <math.h>
#include "Matrix.h"



FCL::FCL(int NeuronsCount, Activation* activation) : Layer(new int[1]{ NeuronsCount }, 1)
{
    this->NeuronsCount = NeuronsCount;
    this->activation = activation;
}

FCL::FCL(int NeuronsCount, Activation* _activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases) : Layer(new int[1]{ NeuronsCount }, 1)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    this->NeuronsCount = NeuronsCount;
    this->activation = _activation;
    Weigths = weights;
    Biases = bias;
}

void FCL::ClearDelta()
{
    Delta->Zero();
    DeltaBiases->Zero();
}

Matrix* FCL::FeedForward(const Matrix* input) 
{
    this->Weigths->CrossProduct(input, this->Result);
    Result->Add(Biases, z);
    activation->FeedForward(z, Result);
    
    return Result;
}

void FCL::Compile(int _previousNeuronsCount)
{
    this->previousNeuronsCount = _previousNeuronsCount;
    if(Weigths == nullptr)
        Weigths = activation->InitWeights(previousNeuronsCount, NeuronsCount);
    if(Delta == nullptr)
        Delta = new Matrix(previousNeuronsCount, NeuronsCount); 
    if(deltaActivation == nullptr)
        deltaActivation = new Matrix(NeuronsCount, 1);
    if(DeltaBiases == nullptr)
        DeltaBiases = new Matrix(NeuronsCount, 1);
    if(Biases == nullptr)
        Biases = activation->InitBiases(NeuronsCount);
    Result = new Matrix(NeuronsCount, 1);
    z = new Matrix(NeuronsCount, 1);
    newDelta = new Matrix(previousNeuronsCount, 1);
}

Matrix* FCL::BackPropagate(const Matrix* lastDelta, const Matrix* PastActivation)
{
    activation->Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);
    
    
    for (int i = 0; i < deltaActivation->getCols() * deltaActivation->getRows(); i++)
    {
        DeltaBiases[0][i] += deltaActivation[0][i];
    }
    

    for (int i = 0; i < previousNeuronsCount; i++)
    {
        newDelta[0][i] = 0;
        for (int j = 0; j < NeuronsCount; j++)
        {
            newDelta[0][i] += deltaActivation[0][j] * Weigths[0][j+i*NeuronsCount];
            Delta[0][i*NeuronsCount+j] += PastActivation[0][i] * deltaActivation[0][j];
        }
    }
    
    
    
    
    
    
    return newDelta;
}

void FCL::UpdateWeights(double learningRate, int batchSize)
{
    double coef = learningRate / batchSize;

    
    for (int i = 0; i < Weigths->getCols() * Weigths->getRows(); i++)
    {
        Weigths[0][i] -= Delta[0][i] * coef;
    }
    for (int i = 0; i < Biases->getCols() * Biases->getRows(); i++)
    {
       Biases[0][i] -= DeltaBiases[0][i] * coef;
    }
    
    Delta->Zero();
    DeltaBiases->Zero();
}

void FCL::UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases)
{

    // Can be optimized by adding all the deltas in the main thread and then updating the weights
    double coef = learningRate / batchSize;
    if(NeuronsCount == 20)
    {
        std::cout << *delta;
    }
    
    for (int i = 0; i < Weigths->getCols() * Weigths->getRows(); i++)
    {
        Weigths[0][i] -= delta[0][i] * coef;
    }
    for (int i = 0; i < Biases->getCols() * Biases->getRows(); i++)
    {
       Biases[0][i] -= deltaBiases[0][i] * coef;
    }
    
    Delta->Zero();
    DeltaBiases->Zero();
    
}

Matrix* FCL::getResult() const
{
    return Result;
}

Matrix* FCL::getDelta() 
{
    return Delta;
}

Matrix* FCL::getDeltaBiases() {
    return DeltaBiases;
}

std::string FCL::getLayerTitle()
{
    std::string buf = "";
    buf += "Layer : Fully Connected Layer\n";
    buf += "Activation Function : " + activation->getName() + "\n";
    buf += "Neurons Count : " + std::to_string(NeuronsCount) + "\n";
    return buf;
}

Layer* FCL::Clone(Matrix* delta, Matrix* deltaBiases)
{
    return new FCL(NeuronsCount, activation, Weigths, Biases, delta, deltaBiases);
}
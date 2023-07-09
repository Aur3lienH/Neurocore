#include "FCL.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include "Matrix.h"
#include "Tools/Serializer.h"
#include "Tools/ManagerIO.h"
#include "LayerShape.h"



FCL::FCL(int NeuronsCount, Activation* activation)
{
    this->NeuronsCount = NeuronsCount;
    this->activation = activation;
    LayerID = 0;
}

FCL::FCL(int NeuronsCount, Activation* _activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    this->NeuronsCount = NeuronsCount;
    this->activation = _activation;
    Weights = weights;
    Biases = bias;
}

void FCL::ClearDelta()
{
    Delta->Zero();
    DeltaBiases->Zero();
}

Matrix* FCL::FeedForward(const Matrix* input) 
{
    input->Flatten();
    this->Weights->CrossProduct(input, Result);
    Result->Add(Biases, z);
    activation->FeedForward(z, Result);
    return Result;
}
void FCL::Compile(LayerShape* previousLayer)
{
    if(previousLayer->size != 1)
    {
        throw std::invalid_argument("Previous Layer must have one dimension ! ");
    }
    previousNeuronsCount = previousLayer->dimensions[0];
    if(Weights == nullptr)
    {
        Weights = activation->InitWeights(previousNeuronsCount, NeuronsCount);
    }
    if(Delta == nullptr)
        Delta = new Matrix(previousNeuronsCount, NeuronsCount); 
    if(deltaActivation == nullptr)
        deltaActivation = new Matrix(NeuronsCount, 1);
    if(DeltaBiases == nullptr)
        DeltaBiases = new Matrix(NeuronsCount, 1);
    if(Biases == nullptr)
        Biases = activation->InitBiases(NeuronsCount);
    if(Result == nullptr)
        Result = new Matrix(NeuronsCount, 1);
    z = new Matrix(NeuronsCount, 1);
    newDelta = new Matrix(previousNeuronsCount, 1);

    layerShape = new LayerShape(NeuronsCount);
    optimizer->Compile(NeuronsCount * previousNeuronsCount + NeuronsCount);
}

const Matrix* FCL::BackPropagate(const Matrix* lastDelta, const Matrix* PastActivation)
{
    newDelta->Flatten();
    activation->Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);
    
    DeltaBiases->Add(deltaActivation,DeltaBiases);
    
    for (int i = 0; i < previousNeuronsCount; i++)
    {
        newDelta[0][i] = 0;
        for (int j = 0; j < NeuronsCount; j++)
        {
            newDelta[0][i] += deltaActivation[0][j] * Weights[0][j + i * NeuronsCount];
            Delta[0][i+j*previousNeuronsCount] += PastActivation[0][i] * deltaActivation[0][j];
        }
    }
    
    return newDelta;
}

void FCL::UpdateWeights(double learningRate, int batchSize)
{
    optimizer->Compute(Delta,Weights);
    optimizer->Compute(DeltaBiases,Biases,Weights->size());
    
    Delta->Zero();
    DeltaBiases->Zero();
}



void FCL::AddDeltaFrom(Layer* otherLayer)
{

    FCL* _FCLLayer = (FCL*)otherLayer;
    for (int i = 0; i < Weights->getCols() * Weights->getRows(); i++)
    {
        Delta[0][i] -= _FCLLayer->Delta[0][i];
    }
    for (int i = 0; i < Biases->getCols() * Biases->getRows(); i++)
    {
       DeltaBiases[0][i] -= _FCLLayer->DeltaBiases[0][i];
    }
    
    Delta->Zero();
    DeltaBiases->Zero();
    
}

const Matrix* FCL::getResult() const
{
    return Result;
}

const Matrix* FCL::getDelta() 
{
    return Delta;
}

const Matrix* FCL::getDeltaBiases() 
{
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

Layer* FCL::Clone()
{
    return new FCL(NeuronsCount, activation, Weights, Biases, nullptr, nullptr);
}

FCL* FCL::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount),sizeof(int));
    Matrix* weigths = Matrix::Read(reader);
    Matrix* biases = Matrix::Read(reader);
    Activation* activation = Activation::Read(reader);
    return new FCL(neuronsCount,activation,weigths,biases,nullptr,nullptr);
}

void FCL::SpecificSave(std::ofstream& write)
{
    write.write(reinterpret_cast<char*>(&NeuronsCount),sizeof(int));
    Weights->Save(write);
    Biases->Save(write);
    activation->Save(write);
}

void FCL::AverageGradients(int batchSize)
{
    Delta->DivideAllDims(batchSize);
    DeltaBiases->DivideAllDims(batchSize);
}



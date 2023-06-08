#include "InputLayer.h"

InputLayer::InputLayer(int inputSize) : Layer(new int[1]{inputSize}, 1)
{
    this->inputSize = inputSize;
    input = new Matrix(inputSize, 1);
}

Matrix* InputLayer::FeedForward(const Matrix* _input) 
{
    for (int i = 0; i < input->getRows() * input->getCols(); i++)
    {
        input[0][i] = _input[0][i];
    }
    return input;
}

void InputLayer::ClearDelta()
{
    
}

Matrix* InputLayer::BackPropagate(const Matrix* delta, const Matrix* lastWeigths)
{
    return nullptr;
}

void InputLayer::Compile(int previousNeuronsCount)
{
}

Matrix* InputLayer::getResult() const
{
    return input;
}

void InputLayer::UpdateWeights(double learningRate, int batchSize)
{
    
}

void InputLayer::UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases)
{
}

std::string InputLayer::getLayerTitle()
{
    std::string buf = "";
    buf += "InputLayer" + '\n';
    buf += "InputSize: " + std::to_string(inputSize) + "\n";
    return buf;
}

Layer* InputLayer::Clone(Matrix* delta, Matrix* deltaBiases)
{
    return new InputLayer(inputSize);
}

Matrix* InputLayer::getDelta()
{
    return nullptr;
}

Matrix* InputLayer::getDeltaBiases()
{
    return nullptr;
}
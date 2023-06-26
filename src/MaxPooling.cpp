#include "MaxPooling.h"


const Matrix* Pooling::FeedForward(const Matrix* input)
{
    return nullptr;
}

Matrix* Pooling::BackPropagate(const Matrix* delta, const Matrix* lastWeigths)
{
    return nullptr;
}

void Pooling::ClearDelta()
{

}

void Pooling::UpdateWeights(double learningRate, int batchSize)
{

}

void Pooling::AddDeltaFrom(Layer* layer)
{

}

void Pooling::Compile(LayerShape* previousOuptut)
{

}

const Matrix* Pooling::getResult() const
{
    return nullptr;
}

std::string Pooling::getLayerTitle()
{
    return std::string();
}

Layer* Pooling::Clone()
{
    return nullptr;
}

void Pooling::SpecificSave(std::ofstream& writer)
{

}

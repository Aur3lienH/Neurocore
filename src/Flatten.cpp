#include "Flatten.h"


Flatten::Flatten()
{

}

void Flatten::Compile(LayerShape* previous)
{
    layerShape = new LayerShape(previous->dimensions[0] * previous->dimensions[1] * previous->dimensions[2]);
}

const Matrix* Flatten::FeedForward(const Matrix* input)
{
    input->Flatten();
    return input;
}

const Matrix* Flatten::BackPropagate(const Matrix* delta, const Matrix* pastActivation)
{
    return delta;
}


void Flatten::UpdateWeights(double learningRate, int batchSize)
{

}


void Flatten::AddDeltaFrom(Layer* layer)
{

}


void Flatten::Compile(LayerShape* layerShape)
{
    this->layerShape = new LayerShape(layerShape->dimensions[0] * layerShape->dimensions[1] * layerShape->dimensions[2]);
}


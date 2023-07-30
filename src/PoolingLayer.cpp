#include "PoolingLayer.h"


void PoolingLayer::ClearDelta()
{

}

PoolingLayer::PoolingLayer(const int filterSize, const int stride) : filterSize(filterSize), stride(stride),
                                                                     result(nullptr), newDelta(nullptr)
{

}

void PoolingLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    // No weights nor biases
}

void PoolingLayer::Compile(LayerShape* previousActivation)
{
    layerShape = new LayerShape((previousActivation->dimensions[0] - filterSize) / stride + 1,
                                (previousActivation->dimensions[1] - filterSize) / stride + 1,
                                previousActivation->dimensions[2]);
    result = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    newDelta = new Matrix(previousActivation->dimensions[0], previousActivation->dimensions[1],
                          previousActivation->dimensions[2]);
}

const Matrix* PoolingLayer::getResult() const
{
    return result;
}

void PoolingLayer::AddDeltaFrom(Layer* layer)
{

}

void PoolingLayer::SpecificSave(std::ofstream& writer)
{
    writer.write((char*) &filterSize, sizeof(int));
    writer.write((char*) &stride, sizeof(int));
}

void PoolingLayer::AverageGradients(const int batchSize)
{

}
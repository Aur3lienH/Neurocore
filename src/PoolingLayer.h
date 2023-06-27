#pragma once
#include "Layer.h"

class PoolingLayer : protected Layer
{
public:
    PoolingLayer(int filterSize, int stride);

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousActivation) override;

    const Matrix* getResult() const override;

    void SpecificSave(std::ofstream& writer) override;


protected:
    const int filterSize, stride;
    Matrix* result;
    Matrix* newDelta;
};
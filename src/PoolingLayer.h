#pragma once

#include "Layer.h"

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int filterSize, int stride);

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousActivation) override;

    [[nodiscard]] const Matrix* getResult() const override;

    void SpecificSave(std::ofstream& writer) override;

    void AverageGradients(int batchSize) override;


protected:
    const int filterSize, stride;
    Matrix* result;
    Matrix* newDelta;
};
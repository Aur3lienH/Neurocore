#pragma once

#include "Layer.cuh"

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int filterSize, int stride);

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousActivation) override;

    [[nodiscard]] const MAT* getResult() const override;

    void SpecificSave(std::ofstream& writer) override;

    void AverageGradients(int batchSize) override;


protected:
    const int filterSize, stride;

    MAT* result;
    MAT* newDelta;
};
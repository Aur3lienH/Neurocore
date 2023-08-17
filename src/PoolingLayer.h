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

#if USE_GPU
    [[nodiscard]] const Matrix_GPU* getResult() const override;
#else
    [[nodiscard]] const Matrix* getResult() const override;
#endif

    void SpecificSave(std::ofstream& writer) override;

    void AverageGradients(int batchSize) override;


protected:
    const int filterSize, stride;

#if USE_GPU
    Matrix_GPU* result;
    Matrix_GPU* newDelta;
#else
    Matrix* result;
    Matrix* newDelta;
#endif
};
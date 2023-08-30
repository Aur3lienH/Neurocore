#pragma once

#include "Layer.cuh"

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int filterSize, int stride);

    ~PoolingLayer() override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousActivation) override;

    [[nodiscard]] const MAT* getResult() const override;

    void SpecificSave(std::ofstream& writer) override;

    void AverageGradients(int batchSize) override;


protected:
    const int filterSize, stride;

public:
    MAT* result;
    MAT* newDelta;

#if USE_GPU
    cudnnPoolingDescriptor_t poolingDescriptor;
    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
#endif
};
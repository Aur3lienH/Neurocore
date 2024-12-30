#pragma once

#include "Layer.cuh"

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int filterSize, int stride);

    ~PoolingLayer();

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void AddDeltaFrom(Layer* layer);

    void Compile(LayerShape* previousActivation);

    [[nodiscard]] const MAT* getResult() const;

    void SpecificSave(std::ofstream& writer);

    void AverageGradients(int batchSize);


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
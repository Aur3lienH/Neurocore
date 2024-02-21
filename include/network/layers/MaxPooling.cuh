#pragma once

#include "PoolingLayer.cuh"
#include "Layer.cuh"

class MaxPoolLayer : public PoolingLayer
{
public:
    MaxPoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const MAT* FeedForward(const MAT* input) override;

    MAT* BackPropagate(const MAT* delta, const MAT* previousActivation) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;

#if USE_GPU
    void Compile(LayerShape* previousActivation) override;
#endif
};
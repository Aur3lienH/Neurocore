#pragma once

#include "PoolingLayer.cuh"
#include "Layer.cuh"

class MaxPoolLayer : public PoolingLayer
{
public:
    MaxPoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const MAT* FeedForward(const MAT* input);

    MAT* BackPropagate(const MAT* delta, const MAT* previousActivation);

    std::string getLayerTitle();

    Layer* Clone();

    void SpecificSave(std::ofstream& writer);

#if USE_GPU
    void Compile(LayerShape* previousActivation);
#endif
};
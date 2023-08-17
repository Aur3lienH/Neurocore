#pragma once

#include "PoolingLayer.h"
#include "Layer.h"

class MaxPoolLayer : public PoolingLayer
{
public:
    MaxPoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

#if USE_GPU

    const Matrix_GPU* FeedForward(const Matrix_GPU* input) override;

    Matrix_GPU* BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* previousActivation) override;

#else

    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) override;

#endif

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;
};
#pragma once
#include "PoolingLayer.h"

class MaxPoolLayer : protected PoolingLayer
{
    MaxPoolLayer(int filterSize, int stride);

    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;
};
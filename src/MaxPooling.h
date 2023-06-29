#pragma once
#include "PoolingLayer.h"
#include "Layer.h"

class MaxPoolLayer : public PoolingLayer
{
public:
    MaxPoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;
};
#pragma once
#include "PoolingLayer.h"

class Pooling : Layer
{
    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousOuptut) override;

    const Matrix* getResult() const override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;

};
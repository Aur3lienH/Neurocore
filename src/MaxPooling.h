#pragma once
#include "PoolingLayer.h"

class MaxPoolLayer : Layer
{
    MaxPoolLayer(int filterSize, int stride);

    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;

    void Compile(LayerShape* previousOutput) override;

    const Matrix* getResult() const override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;

private:
    const int filterSize, stride;
    Matrix* result;
    Matrix* newDelta;
};
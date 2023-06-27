#pragma once
#include "Layer.h"

class Flatten : Layer
{
public:
    Flatten();
    const Matrix* FeedForward(const Matrix* input) override;
    const Matrix* BackPropagate(const Matrix* delta, const Matrix* pastActivation) override;
    void ClearDelta() override;


    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;
    

    void Compile(LayerShape* previousOutput) override;

    const Matrix* getResult() const = 0;



private:
    Matrix* output;



};
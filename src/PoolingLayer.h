#pragma once
#include "Layer.h"

class PoolingLayer : Layer
{
public:
    PoolingLayer();
    virtual Matrix* Compute();


    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);
    void UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases);

    void Compile();

    Matrix* getResult();

    LayerShape* GetLayerShape();

    Matrix* getDelta();
    Matrix* getDeltaBiases();


private:
    int stride;

    Matrix* delta;
    Matrix* deltaBiases;
};
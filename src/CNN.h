#pragma once
#include "vector.h"
#include "Matrix.h"
#include "Layer.h"



class CNN : public Layer
{
public:
    CNN(int* size);
    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    void UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases);
    void Compile(int previousNeuronsCount);
    Matrix* getResult() const;

    std::string getLayerTitle();
    Layer* Clone(Matrix* delta, Matrix* deltaBiases);
    Matrix* getDelta();
    Matrix* getDeltaBiases();

private:
    int* sizes;
    Matrix* filters;
    int* configRow;
    int* configCol;
    int filterCount;
};

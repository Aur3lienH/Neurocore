#pragma once
#include "Matrix.h"
#include "Layer.h"

class InputLayer : public Layer
{
public:
    InputLayer(int inputSize);
    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    void UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases);
    void Compile(int previousNeuronsCount);
    Matrix* getResult() const;
    std::string getLayerTitle();
    Matrix* getDelta();
    Matrix* getDeltaBiases();
    Layer* Clone(Matrix* delta, Matrix* deltaBiases);
    static InputLayer* Load(std::ifstream& reader);
    void SpecificSave(std::ofstream& writer);
private:
    int inputSize;
    Matrix* input = nullptr;
};
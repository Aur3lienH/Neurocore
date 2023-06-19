#pragma once
#include <iostream>
#include <vector>
#include "Tools/Serializer.h"
#include "Matrix.h"


class Layer
{
public:
    Layer(int* NeuronsCount, int NeuronsCountSize);

    virtual Matrix* FeedForward(const Matrix* input) = 0;
    virtual Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths) = 0;
    virtual void ClearDelta() = 0;
    virtual void UpdateWeights(double learningRate, int batchSize) = 0;
    virtual void UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases) = 0;
    virtual void Compile(int previousNeuronsCount) = 0;
    virtual Matrix* getResult() const = 0;
    int getNeuronsCount(int index);
    virtual std::string getLayerTitle() = 0;
    virtual Layer* Clone(Matrix* delta, Matrix* deltaBiases) = 0;
    virtual Matrix* getDelta() = 0;
    virtual Matrix* getDeltaBiases() = 0;

    static Layer* Load(std::ifstream& reader);
    virtual void SpecificSave(std::ofstream& writer) = 0;
    void Save(std::ofstream& writer);

protected:
    int* NeuronsCount;
    int NeuronsCountSize;
    int LayerID;
};


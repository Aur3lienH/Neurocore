#pragma once
#include "FCL.h"
#include "Loss.h"


class LastLayer : public FCL
{
public:
    LastLayer(int NeuronsCount, Activation* activation, Loss* loss);
    LastLayer(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases, Loss* loss);
    double FeedForward(const Matrix* input, const Matrix* desiredOutput);
    void ClearDelta();
    Matrix* BackPropagate(const Matrix* desiredOutput,const Matrix* lastWeights);
    Matrix* getDelta();
    Matrix* getDeltaBiases();
    Layer* Clone();
    double getLossError();
    static LastLayer* Load(std::ifstream& reader);
    void SpecificSave(std::ofstream& writer);
    
private:
    Loss* loss;
    double lossError;
};


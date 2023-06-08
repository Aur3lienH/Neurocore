#include "LastLayer.h"
#include "Activation.h"
#include "Loss.h"

LastLayer::LastLayer(int NeuronsCount, Activation* activation, Loss* _loss) : FCL(NeuronsCount, activation)
{
    this->loss = _loss;
}
LastLayer::LastLayer(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases, Loss* _loss) : FCL(NeuronsCount, activation, weights, bias, delta, deltaBiases)
{
    this->loss = _loss;
}

void LastLayer::ClearDelta()
{
    FCL::ClearDelta();
}

double LastLayer::getLossError()
{
    return lossError;
}

double LastLayer::FeedForward(const Matrix* input, const Matrix* desiredOutput)
{
    Matrix* output = FCL::FeedForward(input);
    return loss->Cost(output, desiredOutput);
    
}

Matrix* LastLayer::BackPropagate(const Matrix* desiredOutput, const Matrix* lastWeights)
{
    loss->CostDerivative(Result, desiredOutput, Result);
    return FCL::BackPropagate(Result, lastWeights);
}


void LastLayer::UpdateWeights(double learningRate, int batchSize)
{
    FCL::UpdateWeights(learningRate, batchSize);
}

void LastLayer::UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaActivation)
{
    FCL::UpdateWeights(learningRate, batchSize, delta, deltaActivation);
}

Layer* LastLayer::Clone(Matrix* delta, Matrix* deltaBiases)
{
    return new LastLayer(NeuronsCount, activation, Weigths, Biases, delta, deltaBiases, loss);
}

Matrix* LastLayer::getDelta()
{
    return Delta;
}

Matrix* LastLayer::getDeltaBiases()
{
    return DeltaBiases;
}



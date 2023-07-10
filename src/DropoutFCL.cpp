//
// Created by mat on 10/07/23.
//

#include "DropoutFCL.h"
#include "InitFunc.h"

void DropoutFCL::SetDropoutRate(double rate)
{
    dropoutRate = rate;

    delete droppedWeights;
    delete droppedBiases;

    droppedWeights = Weights->Copy();
    droppedBiases = Biases->Copy();

    const double scale = 1.0 / (1.0 - dropoutRate);

    // Zero out columns of droppedWeights and droppedBiases and scale the rest
    for (int i = 0; i < droppedWeights->getCols(); i++){
        if (rand() / ((double)RAND_MAX) < dropoutRate){
            for (int j = 0; j < droppedWeights->getRows(); j++){
                (*droppedWeights)(j, i) = 0;
            }
            (*droppedBiases)(0, i) = 0;
        }
        else {
            for (int j = 0; j < droppedWeights->getRows(); j++){
                (*droppedWeights)(j, i) *= scale;
            }
            (*droppedBiases)(0, i) *= scale;
        }
    }
}

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate) : FCL(NeuronsCount, activation), dropoutRate(dropoutRate)
{
    SetDropoutRate(dropoutRate);
}

Matrix* DropoutFCL::FeedForward(const Matrix* input)
{
    input->Flatten();
    (isTraining ? Weights : droppedWeights)->CrossProduct(input, Result);
    Result->Add(Biases, z);
    activation->FeedForward(z, Result);
    return Result;
}

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta,
                       Matrix* deltaActivation, double dropoutRate) : FCL(NeuronsCount, activation, weights, bias, delta, deltaActivation), dropoutRate(dropoutRate)
{
    SetDropoutRate(dropoutRate);
}

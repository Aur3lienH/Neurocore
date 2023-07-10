//
// Created by mat on 10/07/23.
//

#ifndef DEEPLEARNING_DROPOUTFCL_H
#define DEEPLEARNING_DROPOUTFCL_H

#include "FCL.h"
#include "Matrix.h"
#include <random>

class DropoutFCL : public FCL
{
    DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate = .5);
    DropoutFCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaActivation, double dropoutRate = .5);
    Matrix* FeedForward(const Matrix* input) override;
    void SetDropoutRate(double rate);
    void SetIsTraining(const bool isTraining_) { this->isTraining = isTraining_; }
    [[nodiscard]] bool IsTraining() const { return isTraining; }
private:
    Matrix* droppedWeights = nullptr;
    Matrix* droppedBiases = nullptr;
    double dropoutRate = .5;
    bool isTraining = true;
};


#endif //DEEPLEARNING_DROPOUTFCL_H

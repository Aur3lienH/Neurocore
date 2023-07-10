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
    DropoutFCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaActivation, double dropoutRate = .5);
    [[nodiscard]] bool IsTraining() const { return isTraining; }
    void Save();
    void Drop();

public:
    void SetIsTraining(bool isTraining_);
    DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate = .5);

    void Compile(LayerShape* previousLayer) override;

private:
    Matrix* savedWeights = nullptr;
    const double dropoutRate;
    bool isTraining = true;
};


#endif //DEEPLEARNING_DROPOUTFCL_H

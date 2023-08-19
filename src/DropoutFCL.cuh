//
// Created by mat on 10/07/23.
//

#ifndef DEEPLEARNING_DROPOUTFCL_H
#define DEEPLEARNING_DROPOUTFCL_H

#include "FCL.cuh"
#include "Matrix.cuh"
#include <random>

class DropoutFCL : public FCL
{
    DropoutFCL(int NeuronsCount, Activation* activation, MAT* weights, MAT* bias, MAT* delta,
               MAT* deltaActivation, double dropoutRate = .5);

    [[nodiscard]] bool IsTraining() const
    { return isTraining; }

    void Save();

    void Drop();

public:
    void SetIsTraining(bool isTraining_);

    DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate = .5);

    void Compile(LayerShape* previousLayer) override;

private:
    MAT* savedWeights = nullptr;
    const double dropoutRate;
    bool isTraining = true;
};

#endif //DEEPLEARNING_DROPOUTFCL_H

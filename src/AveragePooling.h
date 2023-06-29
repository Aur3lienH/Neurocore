//
// Created by matmu on 27/06/2023.
//

#ifndef DEEPLEARNING_AVERAGEPOOLING_H
#define DEEPLEARNING_AVERAGEPOOLING_H


#include "Layer.h"
#include "PoolingLayer.h"

class AveragePooling : protected PoolingLayer
{
public:
    AveragePooling(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;


private:
    // Filter size squared
    const int fs_2;

};


#endif //DEEPLEARNING_AVERAGEPOOLING_H

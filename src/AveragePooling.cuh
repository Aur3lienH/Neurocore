//
// Created by matmu on 27/06/2023.
//

#ifndef DEEPLEARNING_AVERAGEPOOLING_H
#define DEEPLEARNING_AVERAGEPOOLING_H


#include "Layer.cuh"
#include "PoolingLayer.cuh"

class AveragePoolLayer : public PoolingLayer
{
public:
    AveragePoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const MAT* FeedForward(const MAT* input) override;

    MAT* BackPropagate(const MAT* delta, const MAT* previousActivation) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    void SpecificSave(std::ofstream& writer) override;


private:
    // Filter GetSize squared
    const int fs_2;
};


#endif //DEEPLEARNING_AVERAGEPOOLING_H

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

    const MAT* FeedForward(const MAT* input);

    MAT* BackPropagate(const MAT* delta, const MAT* previousActivation);

    std::string getLayerTitle();

    Layer* Clone();

    void SpecificSave(std::ofstream& writer);

#if USE_GPU

    void Compile(LayerShape* previousActivation) override;

#endif

private:
    // Filter GetSize squared
    const int fs_2;
};


#endif //DEEPLEARNING_AVERAGEPOOLING_H

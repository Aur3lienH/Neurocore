#pragma once

#include <iostream>
#include <vector>
#include "tools/Serializer.h"
#include "matrix/Matrix.cuh"
#include "network/LayerShape.cuh"
#include "network/Optimizers.cuh"

class Layer
{
public:
    Layer();

    virtual ~Layer();

    virtual const MAT* FeedForward(const MAT* input) = 0;

    virtual const MAT* BackPropagate(const MAT* delta, const MAT* previousActivation) = 0;

    [[nodiscard]] virtual const MAT* getResult() const = 0;

    virtual void ClearDelta() = 0;

    virtual void UpdateWeights(double learningRate, int batchSize) = 0;

    virtual void AddDeltaFrom(Layer* layer) = 0;

    virtual void AverageGradients(int batchSize) = 0;

    //Must define the layerShape !
    void Compile(LayerShape* previousOutput, Opti opti);

    virtual void Compile(LayerShape* previousOutput) = 0;

    LayerShape* GetLayerShape();

    virtual std::string getLayerTitle() = 0;

    virtual Layer* Clone() = 0;

    static Layer* Load(std::ifstream& reader);

    virtual void SpecificSave(std::ofstream& writer) = 0;

    void Save(std::ofstream& writer);

protected:
    int LayerID;
    LayerShape* layerShape;
    Optimizer* optimizer = nullptr;
};


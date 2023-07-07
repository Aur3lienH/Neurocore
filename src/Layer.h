#pragma once
#include <iostream>
#include <vector>
#include "Tools/Serializer.h"
#include "Matrix.h"
#include "LayerShape.h"
#include "Optimizers.h"
class Layer
{
public:
    Layer();

    virtual const Matrix* FeedForward(const Matrix* input) = 0;
    virtual const Matrix* BackPropagate(const Matrix* delta, const Matrix* previousActivation) = 0;
    virtual void ClearDelta() = 0;

    
    virtual void UpdateWeights(double learningRate, int batchSize) = 0;
    virtual void AddDeltaFrom(Layer* layer) = 0;

    //Must define the layerShape !
    void Compile(LayerShape* previousOutput, Opti opti);
    virtual void Compile(LayerShape* previousOutput) = 0;
    virtual const Matrix* getResult() const = 0;

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


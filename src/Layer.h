#pragma once
#include <iostream>
#include <vector>
#include "Tools/Serializer.h"
#include "Matrix.h"
#include "LayerShape.h"
class Layer
{
public:
    Layer();

    virtual Matrix* FeedForward(const Matrix* input) = 0;
    virtual Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths) = 0;
    virtual void ClearDelta() = 0;

    
    virtual void UpdateWeights(double learningRate, int batchSize) = 0;
    virtual void AddDeltaFrom(Layer* layer) = 0;


    //Must define the layerShape !
    virtual void Compile(LayerShape* previousOuptut) = 0;
    virtual Matrix* getResult() const = 0;

    LayerShape* GetLayerShape();
    
    virtual std::string getLayerTitle() = 0;
    virtual Layer* Clone() = 0;

    static Layer* Load(std::ifstream& reader);
    virtual void SpecificSave(std::ofstream& writer) = 0;
    void Save(std::ofstream& writer);

protected:
    int LayerID;
    LayerShape* layerShape;
};


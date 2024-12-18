#pragma once

#include <iostream>
#include <vector>
#include "tools/Serializer.h"
#include "matrix/Matrix.cuh"
#include "network/LayerShape.cuh"
#include "network/Optimizers.cuh"

template<typename Derived>
class Layer
{
public:
    Layer();

    virtual ~Layer();

    const MAT* FeedForward(const MAT* input)
    {
	    return static_cast<Derived*>(this)->FeedForwardImpl(input);
    }

    const MAT* BackPropagate(const MAT* delta, const MAT* previousActivation)
    {
	    return static_cast<Derived*>(this)->BackPropagateImpl(delta,previousActivation);
    }

    [[nodiscard]] const MAT* getResult() const
    {
	    return static_cast<Derived*>(this)->getResultImpl();
    }

    void ClearDelta()
    {
	    static_cast<Derived*>(this)->ClearDelta();
    }

    void UpdateWeights(double learningRate, int batchSize)
    {
	    static_cast<Derived*>(this)->UpdateWeights(learningRate, batchSize);	    
    }

    void AddDeltaFrom(Layer* layer)
    {
	    static_cast<Derived*>(this)->AddDeltaFrom(layer);
    }

    void AverageGradients(int batchSize)
    {
	    static_cast<Derived*>(this)->AverageGradients(batchSize);
    }

    //Must define the layerShape !
    void Compile(LayerShape* previousOutput, Opti opti)
    {
	    static_cast<Derived*>(this)->Compile(previousOutput,opti);
    }

    void Compile(LayerShape* previousOutput)
    {
	    static_cast<Derived*>(this)->Compile(previousOutput);
    }

    LayerShape* GetLayerShape()
    {
	    return static_cast<Derived*>(this)->GetLayerShape();
    }

    std::string getLayerTitle();

    Layer* Clone()
    {
	    return static_cast<Derived*>(this)->Clone();
    }

    static Layer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer);

    void Save(std::ofstream& writer);

protected:
    int LayerID;
    LayerShape* layerShape;
    Optimizer* optimizer = nullptr;
};


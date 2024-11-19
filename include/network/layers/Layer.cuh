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
	    return static_cast<Derived*>(this)-><FeedForwardImpl(input);
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
	    static_cast<Derived*>(this)->ClearDeltaImpl();
    }

    void UpdateWeights(double learningRate, int batchSize)
    {
	    static_cast<Derived*>(this)->UpdateWeightsImpl(learningRate, batchSize);	    
    }

    void AddDeltaFrom(Layer* layer)
    {
	    static_cast<Derived*>(this)->AddDeltaFromImpl(layer);
    }

    void AverageGradients(int batchSize)
    {
	    static_cast<Derived*>(this)->AverageGradientsImpl(batchSize);
    }

    //Must define the layerShape !
    void Compile(LayerShape* previousOutput, Opti opti)
    {
	    static_cast<Derived*>(this)->CompileImpl(previousOutput,opti);
    }

    void Compile(LayerShape* previousOutput)
    {
	    static_cast<Derived*>(this)->CompileImpl(previousOutput);
    }

    LayerShape* GetLayerShape()
    {
	    return static_cast<Derived*>(this)->GetLayerShapeImpl();
    }

    std::string getLayerTitle();

    Layer* Clone()
    {
	    return static_cast<Derived*>(this)->CloneImpl();
    }

    static Layer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer);

    void Save(std::ofstream& writer);

protected:
    int LayerID;
    LayerShape* layerShape;
    Optimizer* optimizer = nullptr;
};


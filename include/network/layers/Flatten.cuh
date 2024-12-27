#pragma once

#include "Layer.cuh"

template<LayerShape layershape>
class Flatten
{
public:
    Flatten();

    const MAT* FeedForward(const MAT* _input)
    {
        int first = layershape.x;
    }

    const MAT* BackPropagate(const MAT* delta, const MAT* pastActivation);

    [[nodiscard]] const MAT* getResult() const;

    void ClearDelta();

    static Layer<Flatten>* Load(std::ifstream& reader);

    void UpdateWeights(double learningRate, int batchSize);

    void AddDeltaFrom(Layer<Flatten>* layer);

    template<int x, int y, int z, int size>
    void Compile(LayerShape<x,y,z,size>* previousOutput);

    std::string getLayerTitle();

    void SpecificSave(std::ofstream& writer);

    Layer<Flatten>* Clone();

    void AverageGradients(int batchSize);

private:
    const MAT* input;
    int rows, cols, dims = 0;


};

template<int x, int y, int z, int size>
void Flatten::Compile(LayerShape<x,y,z,size>* previous)
{
    layerShape = new LayerShape<x,y,z,size>(previous->dimensions[0] * previous->dimensions[1] * previous->dimensions[2]);
    rows = previous->dimensions[0];
    cols = previous->dimensions[1];
    dims = previous->dimensions[2];
    LayerID = 3;
}
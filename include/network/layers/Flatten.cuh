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



Flatten::Flatten()
{
    LayerID = 3;
}



const MAT* Flatten::FeedForward(const MAT* _input)
{
    this->input = _input;
    input->Flatten();
    return input;
}

const MAT* Flatten::BackPropagate(const MAT* delta, const MAT* pastActivation)
{
    input->Reshape(rows, cols, dims);
    delta->Reshape(rows, cols, dims);
    return delta;
}

const MAT* Flatten::getResult() const
{
    return input;
}


void Flatten::ClearDelta()
{

}

void Flatten::UpdateWeights(const double learningRate, const int batchSize)
{

}


void Flatten::AddDeltaFrom(Layer* layer)
{

}

std::string Flatten::getLayerTitle()
{
    std::string buffer;
    buffer += "Flatten\n";
    buffer += "Output GetSize : " + std::to_string(layerShape->dimensions[0]) + "\n";
    return buffer;
}


Layer* Flatten::Load(std::ifstream& reader)
{
    return new Flatten();
}


void Flatten::SpecificSave(std::ofstream& writer)
{

}

Layer* Flatten::Clone()
{
    return new Flatten();
}

void Flatten::AverageGradients(const int batchSize)
{

}
#include "network/layers/Flatten.cuh"


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
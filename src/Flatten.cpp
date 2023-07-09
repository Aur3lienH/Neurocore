#include "Flatten.h"


Flatten::Flatten()
{
    LayerID = 3;
}

void Flatten::Compile(LayerShape* previous)
{
    layerShape = new LayerShape(previous->dimensions[0] * previous->dimensions[1] * previous->dimensions[2]);
    rows = previous->dimensions[0];
    cols = previous->dimensions[1];
    dims = previous->dimensions[2];
    LayerID = 3;
}

const Matrix* Flatten::FeedForward(const Matrix* input)
{
    this->input = input;
    input->Flatten();
    return input;
}

const Matrix* Flatten::BackPropagate(const Matrix* delta, const Matrix* pastActivation)
{
    input->Reshape(rows, cols,dims);
    delta->Reshape(rows,cols,dims);
    return delta;
}



void Flatten::ClearDelta()
{

}

void Flatten::UpdateWeights(double learningRate, int batchSize)
{

}


void Flatten::AddDeltaFrom(Layer* layer)
{

}

std::string Flatten::getLayerTitle()
{
    std::string buffer = "";
    buffer += "Flatten\n";
    buffer += "Output Size : " + std::to_string(layerShape->dimensions[0]) + "\n";
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


const Matrix* Flatten::getResult() const
{
    return input;
}

void Flatten::AverageGradients(int batchSize)
{
    
}
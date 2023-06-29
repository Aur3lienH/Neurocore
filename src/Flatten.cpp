#include "Flatten.h"


Flatten::Flatten()
{

}

void Flatten::Compile(LayerShape* previous)
{
    layerShape = new LayerShape(previous->dimensions[0] * previous->dimensions[1] * previous->dimensions[2]);
    rows = previous->dimensions[0];
    cols = previous->dimensions[1];
    dims = previous->dimensions[2];
}

const Matrix* Flatten::FeedForward(const Matrix* input)
{
    this->input = input;
    input->PrintSize();
    std::cout << input->GetOffset();
    input->Flatten();
    input->PrintSize();
    std::cout << input[0][5831];
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


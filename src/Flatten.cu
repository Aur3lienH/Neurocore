#include "Flatten.cuh"


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

#if USE_GPU
const Matrix_GPU* Flatten::FeedForward(const Matrix_GPU* input)
{
    this->input = input;
    input->Flatten();
    return input;
}

const Matrix_GPU* Flatten::BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* pastActivation)
{
    input->Reshape(rows, cols, dims);
    delta->Reshape(rows, cols, dims);
    return delta;
}
const Matrix_GPU* Flatten::getResult() const
{
    return input;
}
#else
const Matrix* Flatten::FeedForward(const Matrix* input)
{
    this->input = input;
    input->Flatten();
    return input;
}

const Matrix* Flatten::BackPropagate(const Matrix* delta, const Matrix* pastActivation)
{
    input->Reshape(rows, cols, dims);
    delta->Reshape(rows, cols, dims);
    return delta;
}
const Matrix* Flatten::getResult() const
{
    return input;
}
#endif


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

void Flatten::AverageGradients(const int batchSize)
{

}
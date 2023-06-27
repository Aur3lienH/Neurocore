#include "LayerShape.h"
#include "InputLayer.h"

InputLayer::InputLayer(int inputSize)
{
    LayerID = 1;
    layerShape = new LayerShape(inputSize);
}

InputLayer::InputLayer(int rows, int cols, int size)
{
    LayerID = 1;
    layerShape = new LayerShape(rows,cols,size);
}

InputLayer::InputLayer(LayerShape* LayerShape)
{
    InputLayer(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
}



const Matrix* InputLayer::FeedForward(const Matrix* _input) 
{
    input = _input;
    return _input;
}


void InputLayer::ClearDelta()
{
    
}

const Matrix* InputLayer::BackPropagate(const Matrix* delta, const Matrix* lastWeigths)
{
    return nullptr;
}

void InputLayer::Compile(LayerShape* layerShape)
{
    
}

const Matrix* InputLayer::getResult() const
{
    return input;
}

void InputLayer::UpdateWeights(double learningRate, int batchSize)
{
    
}

void InputLayer::AddDeltaFrom(Layer* otherLayer)
{

}

std::string InputLayer::getLayerTitle()
{
    std::string buf = "";
    buf += "InputLayer" + '\n';
    //buf += "InputSize: " + std::to_string(inputSize) + "\n";
    return buf;
}

Layer* InputLayer::Clone()
{
    return new InputLayer(layerShape);
}


InputLayer* InputLayer::Load(std::ifstream& reader)
{
    int inputSize;
    reader.read(reinterpret_cast<char*>(&inputSize),sizeof(int));
    return new InputLayer(inputSize);
}

void InputLayer::SpecificSave(std::ofstream& writer) 
{
    //writer.write(reinterpret_cast<char*>(&inputSize),sizeof(int));
}
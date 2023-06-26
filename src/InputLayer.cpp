#include "LayerShape.h"
#include "InputLayer.h"

InputLayer::InputLayer(int inputSize)
{
    input = new Matrix(inputSize, 1);
    LayerID = 1;
    layerShape = new LayerShape(inputSize);
}

InputLayer::InputLayer(int rows, int cols, int size)
{
    LayerID = 1;
    layerShape = new LayerShape(rows,cols,size);
    input = new Matrix[size];
    for (int i = 0; i < size; i++)
    {
        input[i] = Matrix(rows,cols);
    }
}

InputLayer::InputLayer(LayerShape* LayerShape)
{
    InputLayer(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
}



Matrix* InputLayer::FeedForward(const Matrix* _input) 
{
    for (int i = 0; i < layerShape->dimensions[2]; i++)
    {
        for (int j = 0; j < input->getRows() * input->getCols(); j++)
        {
            input[i][j] = _input[i][j];
        }
        
    }
    return input;
}


void InputLayer::ClearDelta()
{
    
}

Matrix* InputLayer::BackPropagate(const Matrix* delta, const Matrix* lastWeigths)
{
    return nullptr;
}

void InputLayer::Compile(LayerShape* layerShape)
{
    
}

Matrix* InputLayer::getResult() const
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

Matrix* InputLayer::getDelta()
{
    return nullptr;
}

Matrix* InputLayer::getDeltaBiases()
{
    return nullptr;
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
#include "LayerShape.h"
#include "InputLayer.h"

InputLayer::InputLayer(const int inputSize)
{
    LayerID = 1;
    layerShape = new LayerShape(inputSize);
}

InputLayer::InputLayer(const int rows, const int cols, const int size)
{
    LayerID = 1;
    layerShape = new LayerShape(rows, cols, size);
}

InputLayer::InputLayer(LayerShape* LayerShape)
{
    LayerID = 1;
    this->layerShape = LayerShape;
}

#if USE_GPU
const Matrix_GPU* InputLayer::FeedForward(const Matrix_GPU* _input)
{
    input = _input;
    return _input;
}

const Matrix_GPU* InputLayer::BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* lastWeights)
{
    return nullptr;
}

const Matrix_GPU* InputLayer::getResult() const
{
    return input;
}
#else

const Matrix* InputLayer::FeedForward(const Matrix* _input)
{
    input = _input;
    return _input;
}

const Matrix* InputLayer::BackPropagate(const Matrix* delta, const Matrix* lastWeights)
{
    return nullptr;
}

const Matrix* InputLayer::getResult() const
{
    return input;
}

#endif

void InputLayer::ClearDelta()
{

}

void InputLayer::Compile(LayerShape* layerShape)
{

}

void InputLayer::UpdateWeights(const double learningRate, const int batchSize)
{

}

void InputLayer::AddDeltaFrom(Layer* otherLayer)
{

}

std::string InputLayer::getLayerTitle()
{
    std::string buf;
    buf += "InputLayer" + '\n';
    buf += layerShape->GetDimensions() + "\n";
    return buf;
}

Layer* InputLayer::Clone()
{
    if (layerShape->size == 1)
    {
        return new InputLayer(layerShape->dimensions[0]);
    }
    return new InputLayer(
            new LayerShape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]));
}


InputLayer* InputLayer::Load(std::ifstream& reader)
{
    LayerShape* layerShape = LayerShape::Load(reader);
    return new InputLayer(layerShape);
}

void InputLayer::SpecificSave(std::ofstream& writer)
{
    layerShape->Save(writer);
}

void InputLayer::AverageGradients(int batchSize)
{

}
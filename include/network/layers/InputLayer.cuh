#pragma once

#include "matrix/Matrix.cuh"
#include "network/layers/Layer.cuh"
#include "network/LayerShape.cuh"

template<int x, int y, int z>
class InputLayer
{
public:
    explicit InputLayer(int inputSize);

    InputLayer(int rows, int cols, int size);

    explicit InputLayer(LayerShape<>* layerShape);

    const MAT* FeedForward(const MAT* input);

    const MAT* BackPropagate(const MAT* delta, const MAT* lastWeights);

    [[nodiscard]] const MAT* getResult() const;

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void AddDeltaFrom(Layer<InputLayer>* otherLayer);

    template<int x, int y, int z, int a>
    void Compile(LayerShape<x,y,z,a>* layerShape);

    std::string getLayerTitle();

    Layer<InputLayer>* Clone();

    static InputLayer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer);

private:
    const MAT* input = nullptr;

    void (* FeedFunc)(const MAT*, Matrix*, int);


    LayerShape* layerShape;
    Optimizer* optimizer = nullptr;
};

template<LayerShape* layershape>
Layer<InputLayer<LayerShape>>::InputLayer(const int inputSize)
{
    layerShape = new LayerShape(inputSize);
}

template<LayerShape* layershape>
Layer<InputLayer<LayerShape>::InputLayer(const int rows, const int cols, const int size)
{
    layerShape = new LayerShape(rows, cols, size);
}

template<LayerShape* layershape>
InputLayer<layershape>::InputLayer(LayerShape* LayerShape)
{
    this->layerShape = LayerShape;
}

template<LayerShape* layershape>
const MAT* InputLayer<LayerShape*>::FeedForward(const MAT* _input)
{
    input = _input;
    return _input;
}

template<LayerShape* layershape>
const MAT* InputLayer<LayerShape*>::BackPropagate(const MAT* delta, const MAT* lastWeights)
{
    return nullptr;
}

template<LayerShape* layershape>
const MAT* InputLayer<LayerShape*>::getResult() const
{
    return input;
}

template<LayerShape* layershape>
void InputLayer<LayerShape*>::ClearDelta()
{

}

template<LayerShape* layershape>
void InputLayer<LayerShape*>::Compile(LayerShape* layerShape)
{
    std::cout << "compiling Input layer\n";
}

template<LayerShape* layershape>
void InputLayer<LayerShape*>::UpdateWeights(const double learningRate, const int batchSize)
{

}

template<LayerShape* layershape>
void InputLayer<LayerShape*>::AddDeltaFrom(Layer* otherLayer)
{

}

template<LayerShape* layershape>
std::string Layer<InputLayer<LayerShape*>>::getLayerTitle()
{
    std::string buf = "InputLayer\n";
    buf += layerShape->GetDimensions() + "\n";
    return buf;
}

template<LayerShape* layershape>
Layer<InputLayer<LayerShape* layershape>> Layer<InputLayer<LayerShape*>>::Clone()
{
    if (layerShape->size == 1)
    {
        return new Layer<new InputLayer(layerShape->dimensions[0])>();
    }
    return new InputLayer(
            new LayerShape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]));
}


template<LayerShape* layershape>
Layer<InputLayer<LayerShape*>>* InputLayer<LayerShape*>::Load(std::ifstream& reader)
{
    LayerShape* layerShape = LayerShape::Load(reader);
    return new InputLayer(layerShape);
}

void Layer<InputLayer<LayerShape*>>::SpecificSave(std::ofstream& writer)
{
    layerShape->Save(writer);
}

template<LayerShape* layershape>
void Layer<InputLayer<LayerShape*>>::AverageGradients(int batchSize)
{

}

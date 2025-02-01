#pragma once

#include "matrix/Matrix.cuh"
#include "network/layers/Layer.cuh"
#include "network/LayerShape.cuh"


template<typename layershape>
class InputLayer final
{

public:
    using Shape = layershape;
public:
    const LMAT<layershape>* FeedForward(const LMAT<layershape>* input);

    const LMAT<layershape>* BackPropagate(const LMAT<layershape>* delta, const LMAT<layershape>* lastWeights);

    [[nodiscard]] const LMAT<layershape>* getResult() const;

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void AddDeltaFrom(InputLayer<layershape>* otherLayer);
    
    void Compile();

    std::string getLayerTitle();

    InputLayer<layershape>* Clone();

    // static InputLayer* Load(std::ifstream& reader);

    // void SpecificSave(std::ofstream& writer);

private:
    const LMAT<layershape>* input = nullptr;

    void (* FeedFunc)(const LMAT<layershape>*, LMAT<layershape>*, int);
};

template<typename layershape>
const LMAT<layershape>* InputLayer<layershape>::FeedForward(const LMAT<layershape>* _input)
{
    input = _input;
    return _input;
}

template<typename layershape>
const LMAT<layershape>* InputLayer<layershape>::BackPropagate(const LMAT<layershape>* delta, const LMAT<layershape>* lastWeights)
{
    return nullptr;
}

template<typename layershape>
const LMAT<layershape>* InputLayer<layershape>::getResult() const
{
    return input;
}

template<typename layershape>
void InputLayer<layershape>::ClearDelta()
{

}

template<typename layershape>
void InputLayer<layershape>::Compile()
{
    //std::cout << "compiling Input layer\n";
}

template<typename layershape>
void InputLayer<layershape>::UpdateWeights(const double learningRate, const int batchSize)
{

}

template<typename layershape>
void InputLayer<layershape>::AddDeltaFrom(InputLayer<layershape>* otherLayer)
{

}

template<typename layershape>
std::string InputLayer<layershape>::getLayerTitle()
{
    std::string buf = "Layer : Input\n";
    buf += layershape::GetDimensions() + "\n";
    return buf;
}

template<typename layershape>
InputLayer<layershape>* InputLayer<layershape>::Clone()
{
    return new InputLayer<layershape>();
}


//template<typename layershape>
//Layer<InputLayer<layershape>>* InputLayer<layershape>::Load(std::ifstream& reader)
//{
//    LayerShape* layerShape = LayerShape::Load(reader);
//    return new InputLayer(layerShape);
//}

//void Layer<InputLayer<layershape>>::SpecificSave(std::ofstream& writer)
//{
//    layerShape->Save(writer);
//}

template<typename layershape>
void InputLayer<layershape>::AverageGradients(int batchSize)
{

}
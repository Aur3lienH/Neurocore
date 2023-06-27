#pragma once
#include "Matrix.h"
#include "Layer.h"
#include "LayerShape.h"

class InputLayer : public Layer
{
public:
    InputLayer(int inputSize);
    InputLayer(int rows, int cols, int size);
    InputLayer(LayerShape* layerShape);


    const Matrix* FeedForward(const Matrix* input);
    const Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    void AddDeltaFrom(Layer* otherLayer);
    void Compile(LayerShape* layerShape);
    const Matrix* getResult() const;
    std::string getLayerTitle();

    Layer* Clone();
    static InputLayer* Load(std::ifstream& reader);
    void SpecificSave(std::ofstream& writer);
private:
    const Matrix* input = nullptr;
    void (*FeedFunc)(const Matrix*,Matrix*,int);
};
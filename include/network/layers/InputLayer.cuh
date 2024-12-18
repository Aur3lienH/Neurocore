#pragma once

#include "matrix/Matrix.cuh"
#include "network/layers/Layer.cuh"
#include "network/LayerShape.cuh"

class InputLayer : public Layer
{
public:
    explicit InputLayer(int inputSize);

    InputLayer(int rows, int cols, int size);

    explicit InputLayer(LayerShape* layerShape);

    const MAT* FeedForward(const MAT* input);

    const MAT* BackPropagate(const MAT* delta, const MAT* lastWeights);

    [[nodiscard]] const MAT* getResult() const;

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void AddDeltaFrom(Layer* otherLayer);

    void Compile(LayerShape* layerShape);

    std::string getLayerTitle();

    Layer* Clone();

    static InputLayer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer);

private:
    const MAT* input = nullptr;

    void (* FeedFunc)(const MAT*, Matrix*, int);
};




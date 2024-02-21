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

    const MAT* FeedForward(const MAT* input) override;

    const MAT* BackPropagate(const MAT* delta, const MAT* lastWeights) override;

    [[nodiscard]] const MAT* getResult() const override;

    void AverageGradients(int batchSize) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* otherLayer) override;

    void Compile(LayerShape* layerShape) override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    static InputLayer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer) override;

private:
    const MAT* input = nullptr;

    void (* FeedFunc)(const MAT*, Matrix*, int);
};
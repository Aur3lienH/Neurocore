#pragma once

#include "Matrix.h"
#include "Layer.h"
#include "LayerShape.h"

class InputLayer : public Layer
{
public:
    explicit InputLayer(int inputSize);

    InputLayer(int rows, int cols, int size);

    explicit InputLayer(LayerShape* layerShape);


    const Matrix* FeedForward(const Matrix* input) override;

    const Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    void AverageGradients(int batchSize) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* otherLayer) override;

    void Compile(LayerShape* layerShape) override;

    [[nodiscard]] const Matrix* getResult() const override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

    static InputLayer* Load(std::ifstream& reader);

    void SpecificSave(std::ofstream& writer) override;

private:
    const Matrix* input = nullptr;

    void (* FeedFunc)(const Matrix*, Matrix*, int);
};
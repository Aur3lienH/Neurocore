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

#if USE_GPU
    const Matrix_GPU* FeedForward(const Matrix_GPU* input) override;

    const Matrix_GPU* BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* lastWeights) override;

    [[nodiscard]] const Matrix_GPU* getResult() const override;
#else
    const Matrix* FeedForward(const Matrix* input) override;

    const Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    [[nodiscard]] const Matrix* getResult() const override;
#endif

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
#if USE_GPU
    const Matrix_GPU* input = nullptr;

    void (* FeedFunc)(const Matrix_GPU*, Matrix_GPU*, int);
#else
    const Matrix* input = nullptr;

    void (* FeedFunc)(const Matrix*, Matrix*, int);
#endif
};
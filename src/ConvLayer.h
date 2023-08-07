#pragma once

#include "Layer.h"
#include "LayerShape.h"
#include "Activation.h"


#include "Matrix.h"


enum Convolution
{
    Valid,
    Same,
    Full
};


class ConvLayer : public Layer
{

public:
    ConvLayer(Matrix* filters, LayerShape* layerShape, Activation* activation);

    ConvLayer(LayerShape* filterShape, Activation* activation);

    ~ConvLayer() override;

    void Compile(LayerShape* previousLayer) override;

    Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    void AddDeltaFrom(Layer* ConvLayer) override;

    void AverageGradients(int batchSize) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void SpecificSave(std::ofstream& writer) override;

    static Layer* Load(std::ifstream& reader);

    [[nodiscard]] Matrix* getResult() const override;

    std::string getLayerTitle() override;

    Layer* Clone() override;

private:
    Matrix* result = nullptr;
    Matrix* rotatedFilter = nullptr;
    Matrix* filters = nullptr;
    //Delta for next layer
    Matrix* delta = nullptr;
    Matrix* preDelta = nullptr;
    Matrix* activationDelta;
    Matrix* z;
    Matrix* previousDeltaMultiplied;
    Matrix* bias;
    Matrix* deltaBias;

    Matrix* nextLayerDelta = nullptr;
    Matrix* nextLayerDeltaTemp = nullptr;

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount = 0;
    uint preivousDimCount = 0;
    uint dimCount = 0;


    LayerShape* filterShape = nullptr;

    Activation* activation = nullptr;


};

#pragma once

#include "Layer.cuh"
#include "LayerShape.cuh"
#include "Activation.cuh"


#include "Matrix.cuh"


enum Convolution
{
    Valid,
    Same,
    Full
};


class ConvLayer : public Layer
{

public:
    ~ConvLayer() override;
    
    ConvLayer(LayerShape* filterShape, Activation* activation);

#if USE_GPU
    ConvLayer(Matrix_GPU* filters, LayerShape* layerShape, Activation* activation);    

    Matrix_GPU* FeedForward(const Matrix_GPU* input) override;

    Matrix_GPU* BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* lastWeights) override;

    [[nodiscard]] Matrix_GPU* getResult() const override;
#else
    ConvLayer(Matrix* filters, LayerShape* layerShape, Activation* activation);    

    Matrix* FeedForward(const Matrix* input) override;

    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    [[nodiscard]] Matrix* getResult() const override;
#endif

    void Compile(LayerShape* previousLayer) override;

    void AddDeltaFrom(Layer* ConvLayer) override;

    void AverageGradients(int batchSize) override;

    void ClearDelta() override;

    void UpdateWeights(double learningRate, int batchSize) override;

    void SpecificSave(std::ofstream& writer) override;

    static Layer* Load(std::ifstream& reader);

    std::string getLayerTitle() override;

    Layer* Clone() override;

private:
#if USE_GPU
    Matrix_GPU* result = nullptr;
    Matrix_GPU* filters = nullptr;
    //Delta for next layer
    Matrix_GPU* delta = nullptr;
    Matrix_GPU* preDelta = nullptr;
    Matrix_GPU* activationDelta;
    Matrix_GPU* z;
    Matrix_GPU* previousDeltaMultiplied;
    Matrix_GPU* bias;
    Matrix_GPU* deltaBias;

    Matrix_GPU* nextLayerDelta = nullptr;
    Matrix_GPU* nextLayerDeltaTemp = nullptr;

    cudnnFilterDescriptor_t filterDesc;
#else
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
#endif

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount = 0;
    uint preivousDimCount = 0;
    uint dimCount = 0;


    LayerShape* filterShape = nullptr;

    Activation* activation = nullptr;


};

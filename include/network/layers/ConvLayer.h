#pragma once

#include "Layer.h"
#include "LayerShape.h"
#include "Matrix.h"
#include "Activation.h"
#include "Operations.h"


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

    ~ConvLayer();

    void Compile(LayerShape* previousLayer);

    Matrix* FeedForward(const Matrix* input);

    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights);

    void AddDeltaFrom(Layer* ConvLayer);

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void SpecificSave(std::ofstream& writer);

    static Layer* Load(std::ifstream& reader);

    [[nodiscard]] Matrix* getResult() const;

    std::string getLayerTitle();

    Layer* Clone();

private:

    void FlipAndCenterFilter();
    void GetOperationsForFullConvolution();

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
    uint offset = 0;


    LayerShape* filterShape = nullptr;

    Activation* activation = nullptr;

    std::vector<Operation*> FullConvOperations = std::vector<Operation*>();
    


};

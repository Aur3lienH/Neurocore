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
    
    void Compile(LayerShape* previousLayer) override;

    Matrix* FeedForward(const Matrix* input) override;
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths) override;

    void AddDeltaFrom(Layer* ConvLayer) override;
    void AverageGradients(int batchSize) override;


    void ClearDelta() override;
    void UpdateWeights(double learningRate, int batchSize) override;
    void SpecificSave(std::ofstream& writer) override;
    static Layer* Load(std::ifstream& reader);
    Matrix* getResult() const override;

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

    
    Matrix* nextLayerDelta = nullptr;

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount = 0;
    uint preivousDimCount = 0;


    LayerShape* filterShape = nullptr;

    Activation* activation = nullptr;
    


};

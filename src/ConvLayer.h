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
    
    void Compile(LayerShape* previousLayer);

    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);

    void AddDeltaFrom(Layer* ConvLayer);
    void AverageGradients(int batchSize);


    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    Layer* Clone();
    void SpecificSave(std::ofstream& writer);
    static Layer* Load(std::ifstream& reader);
    Matrix* getResult() const;

    std::string getLayerTitle();
    Layer* Clone(Matrix* delta, Matrix* deltaBiases);

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

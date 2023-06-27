#pragma once
#include "Layer.h"
#include "LayerShape.h"


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
    explicit ConvLayer(Matrix* filters);
    explicit ConvLayer(LayerShape* filterShape);
    
    void Compile(LayerShape* previousLayer);

    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);

    void AddDeltaFrom(Layer* ConvLayer);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    Layer* Clone();
    void SpecificSave(std::ofstream& writer);
    Matrix* getResult() const;

    std::string getLayerTitle();
    Layer* Clone(Matrix* delta, Matrix* deltaBiases);

private:
    Matrix* result;
    Matrix* rotatedFilter;
    Matrix* filters;
    //Delta for next layer
    Matrix* delta;

    
    Matrix* nextLayerDelta;

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount;
    uint preivousDimCount;
    


    LayerShape* filterShape;



};

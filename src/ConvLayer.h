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
    ConvLayer(Convolution convolution);
    ConvLayer(int* dimesions, int dimensionsNumber, Matrix* filter, Matrix* delta);
    explicit ConvLayer(Matrix* _filter);

    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeigths);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    void UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases);
    void Compile(LayerShape* previousLayer);
    Matrix* getResult() const;

    std::string getLayerTitle();
    Layer* Clone(Matrix* delta, Matrix* deltaBiases);
    Matrix* getDelta();
    Matrix* getDeltaBiases();


private:
    Matrix* result;
    Matrix* rotatedFilter;
    Matrix* filter;
    //Delta for next layer
    Matrix* delta;

    
    Matrix* nextLayerDelta;

    //Result from the previous layer (don't initialize when compiling the layer)
    Matrix* input;
    int outputRows;
    int outputCols;



};

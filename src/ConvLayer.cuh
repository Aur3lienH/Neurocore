#pragma once

#include "Layer.cuh"
#include "LayerShape.cuh"
#include "Activation.cuh"


#include "Matrix.cuh"

class ConvLayer : public Layer
{

public:
    ~ConvLayer() override;

    ConvLayer(LayerShape* filterShape, Activation* activation);

    ConvLayer(MAT* filters, LayerShape* layerShape, Activation* activation);

    MAT* FeedForward(const MAT* input) override;

    MAT* BackPropagate(const MAT* delta, const MAT* prevLayerOutput) override;

    [[nodiscard]] MAT* getResult() const override;

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
    MAT* result = nullptr;
    MAT* rotatedFilter = nullptr;
    MAT* filters = nullptr;
    //Delta for next layer
    MAT* delta = nullptr;
    MAT* preDelta = nullptr;
    MAT* activationDelta;
    MAT* z;
    MAT* previousDeltaMultiplied;
    MAT* bias;
    MAT* deltaBias;

    MAT* nextLayerDelta = nullptr;
    MAT* nextLayerDeltaTemp = nullptr;

#if USE_GPU
    cudnnFilterDescriptor_t filtersDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t forwardAlgo;
    size_t forwardWorkspaceSize = 0;
    float* forwardWorkspace = nullptr;
    cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t backwardDataAlgo;
    size_t backwardFilterWorkspaceSize = 0;
    size_t backwardDataWorkspaceSize = 0;
    float* backwardFilterWorkspace = nullptr;
    float* backwardDataWorkspace = nullptr;

    static inline const int numRequestedConvAlgos = 1;

    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
#endif

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount = 0;
    uint preivousDimCount = 0;
    uint dimCount = 0;


    LayerShape* filterShape = nullptr;

    Activation* activation = nullptr;


};

#include "PoolingLayer.cuh"


void PoolingLayer::ClearDelta()
{

}

PoolingLayer::PoolingLayer(const int filterSize, const int stride) : filterSize(filterSize), stride(stride),
                                                                     result(nullptr), newDelta(nullptr)
{

}

void PoolingLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    // No weights nor biases
}

void PoolingLayer::Compile(LayerShape* previousActivation)
{
    layerShape = new LayerShape((previousActivation->dimensions[0] - filterSize) / stride + 1,
                                (previousActivation->dimensions[1] - filterSize) / stride + 1,
                                previousActivation->dimensions[2]);
    result = new MAT(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    newDelta = new MAT(previousActivation->dimensions[0], previousActivation->dimensions[1],
                       previousActivation->dimensions[2]);

#if USE_GPU

    checkCUDNN(cudnnCreateTensorDescriptor(&forwardInputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardInputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          previousActivation->dimensions[2],
                                          previousActivation->dimensions[0],
                                          previousActivation->dimensions[1]));
    checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          layerShape->dimensions[2],
                                          layerShape->dimensions[0],
                                          layerShape->dimensions[1]));
#endif
}

const MAT* PoolingLayer::getResult() const
{
    return result;
}

void PoolingLayer::AddDeltaFrom(Layer* layer)
{

}

void PoolingLayer::SpecificSave(std::ofstream& writer)
{
    writer.write((char*) &filterSize, sizeof(int));
    writer.write((char*) &stride, sizeof(int));
}

void PoolingLayer::AverageGradients(const int batchSize)
{

}

PoolingLayer::~PoolingLayer()
{
#if USE_GPU
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDescriptor));
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardInputDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardOutputDesc));
#endif
}

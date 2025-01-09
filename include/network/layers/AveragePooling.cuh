#pragma once

#include "Layer.cuh"
#include "PoolingLayer.cuh"

class AveragePoolLayer final
{
public:
    AveragePoolLayer(int filterSize, int stride);

    static Layer* Load(std::ifstream& reader);

    const MAT* FeedForward(const MAT* input);

    MAT* BackPropagate(const MAT* delta, const MAT* previousActivation);

    std::string getLayerTitle();

    Layer* Clone();

    void SpecificSave(std::ofstream& writer);

#if USE_GPU

    void Compile(LayerShape* previousActivation) override;

#endif

private:
    // Filter GetSize squared
    const int fs_2;
};





const MAT* AveragePoolLayer::FeedForward(const MAT* input)
{
#if USE_GPU
    checkCUDNN(
            cudnnPoolingForward(Matrix_GPU::cuda->cudnnHandle,
                                poolingDescriptor,
                                &Matrix_GPU::cuda->one,
                                forwardInputDesc,
                                input->GetData(),
                                &Matrix_GPU::cuda->zero,
                                forwardOutputDesc,
                                result->GetData()));
#else
    Matrix::AveragePool(input, result, filterSize, stride);
#endif

    return result;
}

MAT* AveragePoolLayer::BackPropagate(const MAT* delta, const MAT* previousActivation)
{
#if USE_GPU
    cudnnPoolingBackward(Matrix_GPU::cuda->cudnnHandle,
                         poolingDescriptor,
                         &Matrix_GPU::cuda->one,
                         forwardOutputDesc,
                         result->GetData(),
                         forwardOutputDesc,
                         delta->GetData(),
                         forwardInputDesc,
                         previousActivation->GetData(),
                         &Matrix_GPU::cuda->zero,
                         forwardInputDesc,
                         newDelta->GetData());
#else
    // All elements in the pooling window have the same delta which is delta / (filterSize * filterSize)
    for (int d = 0; d < layerShape->dimensions[2]; ++d)
    {
        for (int i = 0; i < layerShape->dimensions[0]; ++i)
        {
            for (int j = 0; j < layerShape->dimensions[1]; ++j)
            {
                for (int k = 0; k < filterSize; ++k)
                {
                    for (int l = 0; l < filterSize; ++l)
                    {
                        (*newDelta)(i * stride + k, j * stride + l) = (*delta)(i, j) / fs_2;
                    }
                }
            }
        }
        previousActivation->GoToNextMatrix();
        result->GoToNextMatrix();
        newDelta->GoToNextMatrix();
        delta->GoToNextMatrix();
    }

    previousActivation->ResetOffset();
    result->ResetOffset();
    newDelta->ResetOffset();
    delta->ResetOffset();
#endif

    return newDelta;
}


std::string AveragePoolLayer::getLayerTitle()
{
    std::string buf;
    buf += "AveragePool Layer\n";
    buf += "Size: " + std::to_string(filterSize) + "\n";
    buf += "Stride: " + std::to_string(stride) + "\n";

    return buf;
}

Layer* AveragePoolLayer::Clone()
{
    return new AveragePoolLayer(filterSize, stride);
}

AveragePoolLayer::AveragePoolLayer(const int filterSize, const int stride) : PoolingLayer(filterSize, stride),
                                                                             fs_2(filterSize * filterSize)
{

}


Layer* AveragePoolLayer::Load(std::ifstream& reader)
{
    int _filterSize;
    int _tempStride;
    reader.read(reinterpret_cast<char*>(&_filterSize), sizeof(int));
    reader.read(reinterpret_cast<char*>(&_tempStride), sizeof(int));
    return new AveragePoolLayer(_filterSize, _tempStride);
}

void AveragePoolLayer::SpecificSave(std::ofstream& writer)
{
    int tempFilterSize = filterSize;
    int tempStride = stride;
    writer.write(reinterpret_cast<char*>(&tempFilterSize), sizeof(int));
    writer.write(reinterpret_cast<char*>(&tempStride), sizeof(int));
}

#if USE_GPU
void AveragePoolLayer::Compile(LayerShape* previousActivation)
{
    PoolingLayer::Compile(previousActivation);
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDescriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDescriptor,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           filterSize,
                                           filterSize,
                                           0,
                                           0,
                                           stride,
                                           stride));
}
#endif
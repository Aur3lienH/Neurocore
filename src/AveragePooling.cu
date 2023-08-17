//
// Created by matmu on 27/06/2023.
//

#include "AveragePooling.cuh"

#if USE_GPU

const Matrix_GPU* AveragePoolLayer::FeedForward(const Matrix_GPU* input)
{
    Matrix::AveragePool(input, result, filterSize, stride);

    return result;
}

Matrix_GPU* AveragePoolLayer::BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* previousActivation)
{
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

    return newDelta;
}

#else

const Matrix* AveragePoolLayer::FeedForward(const Matrix* input)
{
    Matrix::AveragePool(input, result, filterSize, stride);

    return result;
}

Matrix* AveragePoolLayer::BackPropagate(const Matrix* delta, const Matrix* previousActivation)
{
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

    return newDelta;
}

#endif


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
    reader.read(reinterpret_cast<char*>(_tempStride), sizeof(int));
    return new AveragePoolLayer(_filterSize, _tempStride);
}

void AveragePoolLayer::SpecificSave(std::ofstream& writer)
{
    int tempFilterSize = filterSize;
    int tempStride = stride;
    writer.write(reinterpret_cast<char*>(&tempFilterSize), sizeof(int));
    writer.write(reinterpret_cast<char*>(&tempStride), sizeof(int));
}
//
// Created by matmu on 27/06/2023.
//

#include "AveragePooling.h"

const Matrix* AveragePooling::FeedForward(const Matrix* input)
{
    auto res = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1]);
    Matrix::AveragePool(input, res, filterSize, stride);

    return res;
}

Matrix* AveragePooling::BackPropagate(const Matrix* delta, const Matrix* previousActivation)
{
    // All elements in the pooling window have the same delta which is delta / (filterSize * filterSize)
    for (int i = 0; i < layerShape->dimensions[0]; ++i)
    {
        for (int j = 0; j < layerShape->dimensions[1]; ++j)
        {
            for (int k = 0; k < filterSize; ++k)
            {
                for (int l = 0; l < filterSize; ++l)
                {
                    (*newDelta)(i * stride + k,j * stride + l) = (*delta)(i,j) / fs_2;
                }
            }
        }
    }

    return newDelta;
}

std::string AveragePooling::getLayerTitle()
{
    std::string buf;
    buf += "AveragePool Layer\n";
    buf += "Size: " + std::to_string(filterSize) + "\n";
    buf += "Stride: " + std::to_string(stride) + "\n";

    return buf;
}

Layer* AveragePooling::Clone()
{
    return new AveragePooling(filterSize, stride);
}

AveragePooling::AveragePooling(const int filterSize, const int stride) : PoolingLayer(filterSize, stride),fs_2(filterSize * filterSize)
{

}

#include "MaxPooling.h"


MaxPoolLayer::MaxPoolLayer(int filterSize, int stride) : PoolingLayer(filterSize,stride)
{
    LayerID = 4;
}


const Matrix* MaxPoolLayer::FeedForward(const Matrix* input)
{
    Matrix::MaxPool(input, result, filterSize, stride);
    return result;
}

Matrix* MaxPoolLayer::BackPropagate(const Matrix* delta, const Matrix* previousActivation)
{
    // The idea is that if an element is the maximum than maxPool has selected, then the delta is
    // the same as the previous delta, because the current element is the only one affecting the result.
    previousActivation->PrintSize();
    result->PrintSize();
    newDelta->PrintSize();
    delta->PrintSize();

    for (int m = 0; m < layerShape->dimensions[2]; m++)
    {
        for (int i = 0; i < layerShape->dimensions[0]; ++i)
        {
            for (int j = 0; j < layerShape->dimensions[1]; ++j)
            {
                for (int k = 0; k < filterSize; ++k)
                {
                    for (int l = 0; l < filterSize; ++l)
                    {
                        std::cout << m  << "  " << i << "  " << j << "  " << k << "  " << l << "\n";
                        std::cout << i * stride + k << " : x y : " << j * stride + l << "\n";
                        std::cout << (*previousActivation)(i * stride + k,j * stride + l) << "\n";
                        if ((*previousActivation)(i * stride + k,j * stride + l) == (*result)(i,j))
                            (*newDelta)(i * stride + k,j * stride + l) = (*delta)(i,j);
                        // Should already be 0
                        //else
                        //    (*newDelta)(i * stride + k,j * stride + l) = 0.0;
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

std::string MaxPoolLayer::getLayerTitle()
{
    std::string buf;
    buf += "MaxPool Layer\n";
    buf += "Size: " + std::to_string(filterSize) + "\n";
    buf += "Stride: " + std::to_string(stride) + "\n";

    return buf;
}

Layer* MaxPoolLayer::Clone()
{
    return new MaxPoolLayer(filterSize, stride);
}

Layer* MaxPoolLayer::Load(std::ifstream& reader)
{
    int _filterSize;
    int _tempStride;
    reader.read(reinterpret_cast<char*>(&_filterSize),sizeof(int));
    reader.read(reinterpret_cast<char*>(_tempStride),sizeof(int));
    return new MaxPoolLayer(_filterSize,_tempStride);
}

void MaxPoolLayer::SpecificSave(std::ofstream& writer)
{
    int tempFilterSize = filterSize;
    int tempStride = stride;
    writer.write(reinterpret_cast<char*>(&tempFilterSize),sizeof(int));
    writer.write(reinterpret_cast<char*>(&tempStride),sizeof(int));
}
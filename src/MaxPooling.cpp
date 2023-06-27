#include "MaxPooling.h"


const Matrix* MaxPoolLayer::FeedForward(const Matrix* input)
{
    auto res = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1]);
    Matrix::MaxPool(input, res, filterSize, stride);

    return res;
}

Matrix* MaxPoolLayer::BackPropagate(const Matrix* delta, const Matrix* previousActivation)
{
    // The idea is that if an element is the maximum than maxPool has selected, then the delta is
    // the same as the previous delta, because the current element is the only one affecting the result.
    for (int i = 0; i < layerShape->dimensions[0]; ++i)
    {
        for (int j = 0; j < layerShape->dimensions[1]; ++j)
        {
            for (int k = 0; k < filterSize; ++k)
            {
                for (int l = 0; l < filterSize; ++l)
                {
                    if ((*previousActivation)(i * stride + k,j * stride + l) == (*result)(i,j))
                        (*newDelta)(i * stride + k,j * stride + l) = (*delta)(i,j);
                    // Should already be 0
                    //else
                    //    (*newDelta)(i * stride + k,j * stride + l) = 0.0;
                }
            }
        }
    }

    return newDelta;
}

void MaxPoolLayer::ClearDelta()
{
    // No weights nor biases
}

void MaxPoolLayer::UpdateWeights(double learningRate, int batchSize)
{
    // No weights nor biases
}

void MaxPoolLayer::AddDeltaFrom(Layer* layer)
{

}

void MaxPoolLayer::Compile(LayerShape* previousOutput)
{
    layerShape = new LayerShape(previousOutput->dimensions[0] / stride + 1, previousOutput->dimensions[1] / stride + 1, previousOutput->dimensions[2]);
    result = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1]);
    newDelta = new Matrix(previousOutput->dimensions[0], previousOutput->dimensions[1], (double)0);
}

const Matrix* MaxPoolLayer::getResult() const
{
    return result;
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

void MaxPoolLayer::SpecificSave(std::ofstream& writer)
{
    writer.write((char*)&filterSize, sizeof(int));
    writer.write((char*)&stride, sizeof(int));
}

MaxPoolLayer::MaxPoolLayer(const int filterSize, const int stride) : filterSize(filterSize), stride(stride), result(nullptr), newDelta(nullptr)
{

}

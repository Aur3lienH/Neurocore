//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"
#include "LayerShape.h"


ConvLayer::ConvLayer(Convolution convolution)
{
    LayerID = 3;
    
}


void ConvLayer::Compile(LayerShape* previousLayer)
{
    if(previousLayer->size < 3)
    {
        throw new std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }

    outputCols = previousLayer->dimensions[1] - filter->getCols() + 1;
    outputRows = previousLayer->dimensions[0] - filter->getRows() + 1;

    output = new LayerShape(previousLayer->dimensions[0] - filter->getRows() + 1,previousLayer->dimensions[1] - filter->getCols() + 1, previousLayer->dimensions[2]);
}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    Matrix::Convolution(input,filter,result);
    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta,const Matrix* lastWeights)
{
    Matrix::Flip180(filter,rotatedFilter);
    Matrix::FullConvolution(rotatedFilter,lastDelta,delta);
    Matrix::Convolution(input,lastDelta,nextLayerDelta);
    return nextLayerDelta;
}


void ConvLayer::UpdateWeights(double learningRate, int batchSize)
{
    UpdateWeights(learningRate,batchSize,delta,nullptr);
}

void ConvLayer::UpdateWeights(double learningRate, int batchSize, Matrix* delta, Matrix* deltaBiases)
{
    delta->operator*(learningRate/batchSize);
    filter->Substract(delta,result);
}

void ConvLayer::ClearDelta()
{
    delta->Zero();
}

std::string ConvLayer::getLayerTitle()
{
    std::string buf = "";
    buf += "Convolutional layer";
    return buf;
}

Matrix* ConvLayer::getDelta()
{
    return delta;
}

Matrix* ConvLayer::getResult() const
{
    return result;
}

Matrix* ConvLayer::getDeltaBiases()
{
    return nullptr;
}

Layer* ConvLayer::Clone(Matrix* delta, Matrix* deltaBiases)
{
    
}
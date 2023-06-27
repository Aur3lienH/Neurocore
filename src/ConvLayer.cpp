//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"
#include "InitFunc.h"
#include "LayerShape.h"


ConvLayer::ConvLayer(LayerShape* filterShape)
{
    this->filterShape = filterShape;
}

ConvLayer::ConvLayer(Matrix* filters)
{

}




void ConvLayer::Compile(LayerShape* previousLayer)
{
    if(previousLayer->size < 3)
    {
        throw new std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }  



    int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;


    rotatedFilter = filters->Copy();

    nextLayerDelta = previousLayer->ToMatrix();
    delta = filters->Copy();
    
    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, previousLayer->dimensions[2] * filterCount);


    result = new Matrix(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);


    if(filters == nullptr)
    {
        filters = new Matrix(filterShape->dimensions[0],filterShape->dimensions[1],layerShape->dimensions[2]);
        HeInit(filters->getCols() * filters->getRows() * filters->getDim(),filters);
    }

}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (int i = 0; i < filterCount; i++)
        {
            Matrix::Convolution(input,filters,result);
            filters->GoToNextMatrix();
        }
        input->GoToNextMatrix();
    }
    filters->ResetOffset();
    input->ResetOffset();

    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta,const Matrix* pastActivation)
{

    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint i = 0; i < filterCount; i++)
        {
            Matrix::Flip180(filters,rotatedFilter);
            Matrix::FullConvolution(rotatedFilter,lastDelta,nextLayerDelta);
            Matrix::Convolution(pastActivation,lastDelta,nextLayerDelta);
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            lastDelta->GoToNextMatrix();
        }
        pastActivation->GoToNextMatrix();
    }
    
    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    lastDelta->ResetOffset();
    pastActivation->ResetOffset();
        
    return nextLayerDelta;
}


void ConvLayer::UpdateWeights(double learningRate, int batchSize)
{
    std::cout << *delta;
    delta->MultiplyAllDims(learningRate/batchSize);
    filters->SubstractAllDims(delta,filters);
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


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
    ConvLayer* convLayer = (ConvLayer*)Layer;
    
    for (int i = 0; i < layerShape->dimensions[2]; i++)
    {
        for (int j = 0; j < delta->getRows() * delta->getCols(); j++)
        {
            delta[i][j] += convLayer->delta[i][j];   
        }
    }
}



Matrix* ConvLayer::getResult() const
{
    return result;
}



void ConvLayer::SpecificSave(std::ofstream& writer)
{

}

Layer* ConvLayer::Clone()
{
    return new ConvLayer(this->filters->Copy());
}
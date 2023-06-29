//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"
#include "InitFunc.h"
#include "LayerShape.h"


ConvLayer::ConvLayer(LayerShape* _filterShape)
{
    LayerID = 2;
    filterShape = _filterShape;
}

ConvLayer::ConvLayer(Matrix* filters, LayerShape* filterShape)
{
    LayerID = 2;
    this->filters = filters;
    this->filterShape = filterShape;
}




void ConvLayer::Compile(LayerShape* previousLayer)
{
    if(previousLayer->size < 3)
    {
        throw new std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }

    int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;


    int size = previousLayer->dimensions[2] * filterShape->dimensions[2];

    filterCount = filterShape->dimensions[2];
    preivousDimCount = previousLayer->dimensions[2];

    if(filters == nullptr)
    {
        filters = new Matrix(filterShape->dimensions[0],filterShape->dimensions[1],size);
        HeInit(filters->getCols() * filters->getRows() * filters->getDim(),filters);
    }

    rotatedFilter = filters->Copy();

    nextLayerDelta = new Matrix(previousLayer->dimensions[0],previousLayer->dimensions[1],previousLayer->dimensions[2]);
    delta = filters->Copy();


    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, size);

    result = new Matrix(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    result->Reshape(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (int i = 0; i < filterCount; i++)
        {
            Matrix::Convolution(input,filters,result);
            filters->GoToNextMatrix();
            result->GoToNextMatrix();
        }
        input->GoToNextMatrix();
    }
    filters->ResetOffset();
    input->ResetOffset();
    result->ResetOffset();

    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta,const Matrix* pastActivation)
{
    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint j = 0; j < filterCount; j++)
        {
            Matrix::Flip180(filters,rotatedFilter);
            Matrix::FullConvolution(rotatedFilter,lastDelta,nextLayerDelta);
            Matrix::Convolution(pastActivation,lastDelta,delta);
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
    buf += "Convolutional layer\n";
    buf += "Filter count per channel : " + std::to_string(filterShape->dimensions[2]) + "\n";
    buf += "Feature map count : " + std::to_string(layerShape->dimensions[2]) + "\n";
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
    filters->Save(writer);
    filterShape->Save(writer);
}

Layer* ConvLayer::Load(std::ifstream& reader)
{
    Matrix* filters = Matrix::Read(reader);
    LayerShape* filterShape = LayerShape::Load(reader);
    return new ConvLayer(filters,filterShape);
}

Layer* ConvLayer::Clone()
{
    return new ConvLayer(this->filters->Copy(), new LayerShape(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]));
}
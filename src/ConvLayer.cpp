//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"
#include "LayerShape.h"


ConvLayer::ConvLayer(Matrix* _filter)
{
    LayerID = 3;
    filter = _filter;
}




void ConvLayer::Compile(LayerShape* previousLayer)
{
    if(previousLayer->size < 3)
    {
        throw new std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }  

    int outputRow = previousLayer->dimensions[0] - filter->getRows() + 1;
    int outputCol = previousLayer->dimensions[1] - filter->getCols() + 1;

    result = new Matrix[previousLayer->dimensions[2]];
    for (int i = 0; i < previousLayer->dimensions[2]; i++)
    {
        result[i] = Matrix(outputRow,outputCol);
    }

    rotatedFilter = filter->Copy();

    nextLayerDelta = previousLayer->ToMatrix();
    delta = filter->Copy();
    
    layerShape = new LayerShape(previousLayer->dimensions[0] - filter->getRows() + 1,previousLayer->dimensions[1] - filter->getCols() + 1, previousLayer->dimensions[2]);
}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    this->input = input;
    Matrix::Convolution(input,filter,result);
    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta,const Matrix* lastWeights)
{
    Matrix::Flip180(filter,rotatedFilter);
    Matrix::FullConvolution(rotatedFilter,lastDelta,nextLayerDelta);
    Matrix::Convolution(input,lastDelta,nextLayerDelta);
    return nextLayerDelta;
}


void ConvLayer::UpdateWeights(double learningRate, int batchSize)
{
    std::cout << *delta;
    delta->operator*(learningRate/batchSize);
    filter->Substract(delta,filter);
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
    return new ConvLayer(this->filter->Copy());
}
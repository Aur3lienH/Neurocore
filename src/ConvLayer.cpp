//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"
#include "InitFunc.h"
#include "LayerShape.h"
#include "Optimizers.h"

ConvLayer::ConvLayer(LayerShape* _filterShape, Activation* activation)
{
    LayerID = 2;
    filterShape = _filterShape;
    this->activation = activation;
}

ConvLayer::ConvLayer(Matrix* filters, LayerShape* filterShape, Activation* activation)
{
    LayerID = 2;
    this->filters = filters;
    this->filterShape = filterShape;
    this->activation = activation;
}


void ConvLayer::Compile(LayerShape* previousLayer)
{
    if (previousLayer->size < 3)
    {
        throw std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }

    if (activation == nullptr)
    {
        throw std::invalid_argument("ConvLayer : Must have an activation function !");
    }

    int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;


    std::cout << previousLayer->dimensions[2] << " wow this is size !\n";

    filterCount = filterShape->dimensions[2];
    preivousDimCount = previousLayer->dimensions[2];

    if (filters == nullptr)
    {
<<<<<<< HEAD
        filters = new Matrix(filterShape->dimensions[0],filterShape->dimensions[1],filterShape->dimensions[2]);
        HeInit(1,filters);
=======
        filters = new Matrix(filterShape->dimensions[0], filterShape->dimensions[1], size);
        HeInit(1, filters);
>>>>>>> 1e3c2f41d7880c92061d2c355d78715ae9f80f34
    }

    rotatedFilter = filters->Copy();

    nextLayerDelta = new Matrix(previousLayer->dimensions[0], previousLayer->dimensions[1],
                                previousLayer->dimensions[2]);
    delta = filters->Copy();
    preDelta = filters->Copy();


<<<<<<< HEAD
    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, filterShape->dimensions[2]);
=======
    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,
                                previousLayer->dimensions[1] - filters->getCols() + 1, size);
>>>>>>> 1e3c2f41d7880c92061d2c355d78715ae9f80f34

    result = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    activationDelta = result->Copy();

    optimizer->Compile(filters->size());


}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    //result->PrintSize();
    for (uint j = 0; j < preivousDimCount; j++)
    {
        for (int i = 0; i < filterCount; i++)
        {
            Matrix::Convolution(input, filters, z);
            filters->GoToNextMatrix();
            z->GoToNextMatrix();
        }
        input->GoToNextMatrix();
    }
    filters->ResetOffset();
    input->ResetOffset();
    z->ResetOffset();
    activation->FeedForward(z, result);
    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta, const Matrix* lastWeights)
{

    activation->Derivative(z, activationDelta);
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);

    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint j = 0; j < filterCount; j++)
        {
            Matrix::Flip180(filters, rotatedFilter);
            Matrix::FullConvolution(rotatedFilter, previousDeltaMultiplied, nextLayerDelta);
            Matrix::Convolution(lastWeights, previousDeltaMultiplied, preDelta);
            delta->AddAllDims(preDelta, delta);
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
        }
        lastWeights->GoToNextMatrix();
    }


    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    lastWeights->ResetOffset();

    return nextLayerDelta;
}


void ConvLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);
}


void ConvLayer::ClearDelta()
{
    delta->Zero();
}

std::string ConvLayer::getLayerTitle()
{
    std::string buf;
    buf += "Convolutional layer\n";
    buf += "Filter count per channel : " + std::to_string(filterShape->dimensions[2]) + "\n";
    buf += "Feature map count : " + std::to_string(layerShape->dimensions[2]) + "\n";
    buf += "Output size : " + layerShape->GetDimensions() + "\n";
    return buf;
}


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
}


Matrix* ConvLayer::getResult() const
{
    return result;
}


void ConvLayer::SpecificSave(std::ofstream& writer)
{
    filters->Save(writer);
    filterShape->Save(writer);
    activation->Save(writer);
}

Layer* ConvLayer::Load(std::ifstream& reader)
{
    Matrix* filters = Matrix::Read(reader);
    LayerShape* filterShape = LayerShape::Load(reader);
    Activation* activation = Activation::Read(reader);
    return new ConvLayer(filters, filterShape, activation);
}

Layer* ConvLayer::Clone()
{
    auto* filterCopy = filters->CopyWithSameData();
    return new ConvLayer(filterCopy, new LayerShape(layerShape->dimensions[0], layerShape->dimensions[1],
                                                    layerShape->dimensions[2]), activation);
}

void ConvLayer::AverageGradients(int batchSize)
{
    delta->DivideAllDims(batchSize);
}

ConvLayer::~ConvLayer()
{
    delete filters;
    delete filterShape;
    delete activation;
    delete result;
    delete z;
    delete delta;
    delete preDelta;
    delete previousDeltaMultiplied;
    delete activationDelta;
    delete nextLayerDelta;
    delete rotatedFilter;
}




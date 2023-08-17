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

#if USE_GPU
ConvLayer::ConvLayer(Matrix_GPU* filters, LayerShape* filterShape, Activation* activation)
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



    filterCount = filterShape->dimensions[2];
    preivousDimCount = previousLayer->dimensions[2];
    dimCount = filterCount * preivousDimCount;
    

    if (filters == nullptr)
    {
        filters = new Matrix_GPU(filterShape->dimensions[0],filterShape->dimensions[1],(int)dimCount);
        HeInit(1,filters);
    }

    rotatedFilter = filters->Copy();

    nextLayerDelta = new Matrix(previousLayer->dimensions[0], previousLayer->dimensions[1],previousLayer->dimensions[2]);

    nextLayerDeltaTemp = new Matrix(previousLayer->dimensions[0],previousLayer->dimensions[1]);


    delta = filters->Copy();
    preDelta = new Matrix_GPU(filters->getRows(),filters->getCols());


    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, dimCount);

    result = new Matrix_GPU(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    activationDelta = result->Copy();

    bias = new Matrix_GPU(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
    deltaBias = new Matrix_GPU(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);

    optimizer->Compile(filters->size() + bias->size());

    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
}
Matrix_GPU* ConvLayer::FeedForward(const Matrix_GPU* input)
{
    result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    //result->PrintSize();
    for (uint j = 0; j < preivousDimCount; j++)
    {
        for (int i = 0; i < filterCount; i++)
        {
            Matrix::Convolution(input, filters, z);
            filters->GoToNextMatrix();
        }
        input->GoToNextMatrix();
    }
    z->AddAllDims(bias,z);
    filters->ResetOffset();
    input->ResetOffset();
    activation->FeedForward(z, result);
    return result;
}


//May be optimized by not rotating the matrix
Matrix_GPU* ConvLayer::BackPropagate(const Matrix_GPU* lastDelta, const Matrix_GPU* lastWeights)
{

    activation->Derivative(z, activationDelta);
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);

    deltaBias->AddAllDims(previousDeltaMultiplied,deltaBias);

    nextLayerDelta->Zero();

    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint j = 0; j < filterCount; j++)
        {
            Matrix::Flip180(filters, rotatedFilter);
            Matrix::FullConvolution(rotatedFilter, previousDeltaMultiplied, nextLayerDeltaTemp);
            nextLayerDelta->Add(nextLayerDeltaTemp,nextLayerDelta);
            Matrix::Convolution(lastWeights, previousDeltaMultiplied, preDelta);
            delta->Add(preDelta,delta);
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
        }
        lastWeights->GoToNextMatrix();
        nextLayerDelta->GoToNextMatrix();
    }

    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    lastWeights->ResetOffset();
    nextLayerDelta->ResetOffset();

    return nextLayerDelta;
}

 
void ConvLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);
    optimizer->Compute(deltaBias,bias,filters->size());
}

Matrix_GPU* ConvLayer::getResult() const
{
    return result;
}


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
}

Layer* ConvLayer::Load(std::ifstream& reader)
{
    Matrix* filters_CPU = Matrix::Read(reader);
    Matrix_GPU* filters = new Matrix_GPU(*filters_CPU);
    LayerShape* filterShape = LayerShape::Load(reader);
    Activation* activation = Activation::Read(reader);
    return new ConvLayer(filters, filterShape, activation);
}
#else
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



    filterCount = filterShape->dimensions[2];
    preivousDimCount = previousLayer->dimensions[2];
    dimCount = filterCount * preivousDimCount;
    

    if (filters == nullptr)
    {
        filters = new Matrix(filterShape->dimensions[0],filterShape->dimensions[1],(int)dimCount);
        HeInit(1,filters);
    }

    rotatedFilter = filters->Copy();

    nextLayerDelta = new Matrix(previousLayer->dimensions[0], previousLayer->dimensions[1],previousLayer->dimensions[2]);

    nextLayerDeltaTemp = new Matrix(previousLayer->dimensions[0],previousLayer->dimensions[1]);


    delta = filters->Copy();
    preDelta = new Matrix(filters->getRows(),filters->getCols());


    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, dimCount);

    result = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    activationDelta = result->Copy();

    bias = new Matrix(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);
    deltaBias = new Matrix(layerShape->dimensions[0],layerShape->dimensions[1],layerShape->dimensions[2]);

    optimizer->Compile(filters->size() + bias->size());
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
        }
        input->GoToNextMatrix();
    }
    z->AddAllDims(bias,z);
    filters->ResetOffset();
    input->ResetOffset();
    activation->FeedForward(z, result);
    return result;
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta, const Matrix* lastWeights)
{

    activation->Derivative(z, activationDelta);
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);

    deltaBias->AddAllDims(previousDeltaMultiplied,deltaBias);

    nextLayerDelta->Zero();

    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint j = 0; j < filterCount; j++)
        {
            Matrix::Flip180(filters, rotatedFilter);
            Matrix::FullConvolution(rotatedFilter, previousDeltaMultiplied, nextLayerDeltaTemp);
            nextLayerDelta->Add(nextLayerDeltaTemp,nextLayerDelta);
            Matrix::Convolution(lastWeights, previousDeltaMultiplied, preDelta);
            delta->Add(preDelta,delta);
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
        }
        lastWeights->GoToNextMatrix();
        nextLayerDelta->GoToNextMatrix();
    }

    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    lastWeights->ResetOffset();
    nextLayerDelta->ResetOffset();

    return nextLayerDelta;
}

 
void ConvLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);
    optimizer->Compute(deltaBias,bias,filters->size());
}

Matrix* ConvLayer::getResult() const
{
    return result;
}


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
}

Layer* ConvLayer::Load(std::ifstream& reader)
{
    Matrix* filters = Matrix::Read(reader);
    LayerShape* filterShape = LayerShape::Load(reader);
    Activation* activation = Activation::Read(reader);
    return new ConvLayer(filters, filterShape, activation);
}
#endif

void ConvLayer::ClearDelta()
{
    delta->Zero();
    deltaBias->Zero();
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


void ConvLayer::SpecificSave(std::ofstream& writer)
{
    filters->Save(writer);
    filterShape->Save(writer);
    activation->Save(writer);
}

Layer* ConvLayer::Clone()
{
    auto* filterCopy = filters->CopyWithSameData();
    return new ConvLayer(filterCopy, new LayerShape(filterShape->dimensions[0], filterShape->dimensions[1],
                                                    filterShape->dimensions[2]), activation);
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
#if USE_GPU
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
#else
    delete rotatedFilter;
#endif
}




//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.cuh"
#include "InitFunc.cuh"
#include "LayerShape.cuh"

ConvLayer::ConvLayer(LayerShape* _filterShape, Activation* activation)
{
    LayerID = 2;
    filterShape = _filterShape;
    this->activation = activation;
}


ConvLayer::ConvLayer(MAT* filters, LayerShape* filterShape, Activation* activation)
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

    //const int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    //const int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;


    filterCount = filterShape->dimensions[2];
    preivousDimCount = previousLayer->dimensions[2];
    dimCount = filterCount * preivousDimCount;


    if (filters == nullptr)
    {
        filters = new MAT(filterShape->dimensions[0], filterShape->dimensions[1], (int) dimCount);
        HeInit(1, filters);
    }

#if not USE_GPU
    rotatedFilter = filters->Copy();
#endif

    nextLayerDelta = new MAT(previousLayer->dimensions[0], previousLayer->dimensions[1],
                             previousLayer->dimensions[2]);

    nextLayerDeltaTemp = new MAT(previousLayer->dimensions[0], previousLayer->dimensions[1]);


    delta = filters->Copy();
    preDelta = new MAT(filters->GetRows(), filters->GetCols());


    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->GetRows() + 1, previousLayer->dimensions[1] -
                                                                                       filters->GetCols() + 1,
                                dimCount);

    result = new MAT(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    activationDelta = result->Copy();

    bias = new MAT(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    deltaBias = new MAT(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    optimizer->Compile(filters->GetSize() + bias->GetSize());

#if USE_GPU
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
#endif
}

MAT* ConvLayer::FeedForward(const MAT* input)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::FeedForward is not implmentedd on GPU");
#else
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
    z->AddAllDims(bias, z);
    filters->ResetOffset();
    input->ResetOffset();
    activation->FeedForward(z, result);
    return result;

#endif
}


//May be optimized by not rotating the matrix
MAT* ConvLayer::BackPropagate(const MAT* lastDelta, const MAT* lastWeights)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::BackPropagate is not implemented on GPU");
#else
    activation->Derivative(z, activationDelta);
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);

    deltaBias->AddAllDims(previousDeltaMultiplied, deltaBias);

    nextLayerDelta->Zero();

    for (uint i = 0; i < preivousDimCount; i++)
    {
        for (uint j = 0; j < filterCount; j++)
        {
            Matrix::Flip180(filters, rotatedFilter);
            Matrix::FullConvolution(rotatedFilter, previousDeltaMultiplied, nextLayerDeltaTemp);
            nextLayerDelta->Add(nextLayerDeltaTemp, nextLayerDelta);
            Matrix::Convolution(lastWeights, previousDeltaMultiplied, preDelta);
            delta->Add(preDelta, delta);
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
#endif
}


void ConvLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);
    optimizer->Compute(deltaBias, bias, filters->GetSize());
}

MAT* ConvLayer::getResult() const
{
    return result;
}


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::AddDeltaFrom is not implemented on GPU");
#else
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
    deltaBias->AddAllDims(convLayer->deltaBias, deltaBias);
#endif
}

Layer* ConvLayer::Load(std::ifstream& reader)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::Load is not implmentedd on GPU");
#else
    Matrix* filters = Matrix::Read(reader);
    LayerShape* filterShape = LayerShape::Load(reader);
    Activation* activation = Activation::Read(reader);
    return new ConvLayer(filters, filterShape, activation);
#endif
}

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
    buf += "Output GetSize : " + layerShape->GetDimensions() + "\n";
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




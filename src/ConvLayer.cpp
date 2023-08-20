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
    //Check if the previous layer has 3 dimensions, if not throw an error
    if (previousLayer->size < 3)
    {
        throw std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }

    //If there is no activation function, throw an error
    if (activation == nullptr)
    {
        throw std::invalid_argument("ConvLayer : Must have an activation function !");
    }


    //Calculate the dimensions of the actual layer
    int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;


    //Number of filter per channel
    filterCount = filterShape->dimensions[2];
    //Number of channel in the previous layer
    preivousDimCount = previousLayer->dimensions[2];

    //Number of dimCount
    dimCount = filterCount * preivousDimCount;

    //If the filters has no been initialized, create it and initialize it with random values
    if (filters == nullptr)
    {
        filters = new Matrix(filterShape->dimensions[0],filterShape->dimensions[1],(int)dimCount);
        //Function to init the filters with random values
        WeightsInit::HeUniform(filterShape->dimensions[0] * filterShape->dimensions[1] ,filters);
    }


    layerShape = new LayerShape(previousLayer->dimensions[0] - filters->getRows() + 1,previousLayer->dimensions[1] - filters->getCols() + 1, dimCount);
    
    rotatedFilter = new Matrix(filters->getRows(),filters->getCols(),filters->getDim(),false);
    nextLayerDelta = new Matrix(previousLayer->dimensions[0], previousLayer->dimensions[1],previousLayer->dimensions[2]);
    nextLayerDeltaTemp = new Matrix(previousLayer->dimensions[0], previousLayer->dimensions[1],previousLayer->dimensions[2]);
    delta = filters->Copy();
    delta->Zero();
    preDelta = new Matrix(filters->getRows(),filters->getCols());
    result = new Matrix(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);

    z = result->Copy();
    previousDeltaMultiplied = result->Copy();
    offset = previousDeltaMultiplied->getRows() - 1;
    activationDelta = result->Copy();

    bias = new Matrix(1,1,(int)dimCount,false);

    for (int i = 0; i < bias->size(); i++)
    {
        (*bias)[i] = 0.01;
    }

    deltaBias = new Matrix(1,1,(int)dimCount,false);

    optimizer->Compile(filters->size() + bias->size());

    
    for (int j = 0; j < preivousDimCount; j++)
    {
        for (int i = 0; i < filterCount; i++)
        {
            GetOperationsForFullConvolution();
            filters->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
        }
        nextLayerDelta->GoToNextMatrix();
    }
    filters->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    nextLayerDelta->ResetOffset();
    

    std::cout << "number of operations : " << FullConvOperations.size() << " \n";

    std::cout << "compiled !\n";


}

Matrix* ConvLayer::FeedForward(const Matrix* input)
{
    //Reshape the layer in case it does not have the right shape
    result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    //result->PrintSize();
    //Loop through all the dimensions of the previous layer
    for (uint i = 0; i < preivousDimCount; i++)
    {
        //Loop through all the dimensions of the actual layer
        for (int j = 0; j < filterCount; j++)
        {
            //Apply convolution between input and filters and output it in z
            Matrix::Convolution(input, filters, z);

            //Add the bias to the result
            for (int k = 0; k < z->getRows() * z->getCols(); k++)
            {
                (*z)[k] = bias[0][0] + (*z)[k];
            }
            
            //Filters and bias are moved to the next matrix
            filters->GoToNextMatrix();
            bias->GoToNextMatrix();
            z->GoToNextMatrix();
        }
        //Input is moved to the next matrix
        input->GoToNextMatrix();
    }
    //All the matrix offset are reset
    filters->ResetOffset();
    input->ResetOffset();
    bias->ResetOffset();
    z->ResetOffset();

    //Apply activation function on all the matrix
    activation->FeedForward(z, result);

    return result;
}

void ConvLayer::FlipAndCenterFilter()
{
    for (int d = 0; d < filters->getDim(); d++)
    {
        for (int i = 0; i < filters->getCols(); ++i)
        {
            for (int j = 0; j < filters->getRows(); ++j)
            {
                (*rotatedFilter)(i + offset, j + offset) = (*filters)(filters->getRows() - 1 - j, filters->getCols() - 1 - i);
            }
        }
        rotatedFilter->GoToNextMatrix();
        filters->GoToNextMatrix();
    }

    rotatedFilter->ResetOffset();
    filters->ResetOffset();
    
}


//May be optimized by not rotating the matrix
Matrix* ConvLayer::BackPropagate(const Matrix* lastDelta, const Matrix* input)
{
    //Set to zero the delta of the next layer
    nextLayerDelta->Zero();

    //Calculate the partial derivative of the activation function
    activation->Derivative(z, activationDelta);

    //Multiply the partial derivative of the activation function with the partial derivative of the previous layer
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);

    for (int k = 0; k < FullConvOperations.size(); k++)
    {
        FullConvOperations[k]->Compute();
    }

    //Loop through all the dimensions of the previous layer
    for (uint i = 0; i < preivousDimCount; i++)
    {
        //Loop through all the dimensions of the actual layer
        for (uint j = 0; j < filterCount; j++)
        {
            //Flip the filter
            Matrix::Flip180(filters,rotatedFilter);
            
            //Calculate the partial derivative for the previous layer
            //Matrix::FullConvolution(rotatedFilter,previousDeltaMultiplied,nextLayerDeltaTemp);

            //Accumulate the result of the partial derivative
            //nextLayerDelta->Add(nextLayerDeltaTemp,nextLayerDelta);

            

            //Calculate the partial derivative of the weights
            Matrix::Convolution(input, previousDeltaMultiplied, preDelta);
            
            //Accumulate the result
            delta->Add(preDelta,delta);

            //Filters, rotatedFilter, previousDeltaMultiplied and delta are moved to the next matrix
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
            delta->GoToNextMatrix();
        }
        // Input and nextLayerDelta are moved to the next matrix
        input->GoToNextMatrix();
        nextLayerDelta->GoToNextMatrix();
    }
    //Resetting all the matrix offset
    nextLayerDelta->ResetOffset();
    delta->ResetOffset();
    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    input->ResetOffset();


    //Return the partial derivative for the previous layer
    return nextLayerDelta;
}

 
void ConvLayer::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);
    

    for (int i = 0; i < deltaBias->getDim(); i++)
    {
        for (int j = 0; j < delta->getRows() * delta->getCols(); j++)
        {
            (*deltaBias)[0] += (*delta)[j];
        }
        deltaBias->GoToNextMatrix();
        delta->GoToNextMatrix();
    }

    deltaBias->ResetOffset();
    delta->ResetOffset();
    

    optimizer->Compute(deltaBias,bias,bias->size());
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
    buf += "Output size : " + layerShape->GetDimensions() + "\n";
    return buf;
}


void ConvLayer::AddDeltaFrom(Layer* Layer)
{
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
    deltaBias->AddAllDims(convLayer->deltaBias,deltaBias);
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

void ConvLayer::GetOperationsForFullConvolution()
{
    const int outputCols = previousDeltaMultiplied->getCols() + filters->getCols() - 1;
    const int outputRows = previousDeltaMultiplied->getRows() + filters->getRows() - 1;

    const int filterCols = filters->getCols();
    const int filterRows = filters->getRows();

    const int inputCols = previousDeltaMultiplied->getCols();
    const int inputRows = previousDeltaMultiplied->getRows();

    const int r = filterRows - 1;
    const int c = filterCols - 1;

    
    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            for (int k = 0; k < filterCols; k++)
            {
                float* filterPointer = nullptr;
                float* matrixPointer = nullptr;
                int operationNumber = 0;
                for (int l = 0; l < filterRows; l++)
                {
                    const int inputRow = i - k;
                    const int inputCol = j - l;
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                    {
                        float* filterPointer = &((*rotatedFilter)(k, l));
                        float* matrixPointer = &((*previousDeltaMultiplied)(inputRow, inputCol));
                        FullConvOperations.push_back(new MulAddTo1(filterPointer, matrixPointer, &((*nextLayerDelta)(i, j)), 1));
                    }
                }
                
            }
        }
    }
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
    deltaBias->DivideAllDims(batchSize);
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




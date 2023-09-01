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
    checkCUDNN(cudnnCreateFilterDescriptor(&filtersDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filtersDesc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          filterShape->dimensions[2],
                                          1,
                                          filterShape->dimensions[0],
                                          filterShape->dimensions[1]));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               1,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));


    checkCUDNN(cudnnCreateTensorDescriptor(&forwardInputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardInputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          previousLayer->dimensions[0],
                                          previousLayer->dimensions[1]));

    checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          filterShape->dimensions[2],
                                          layerShape->dimensions[0],
                                          layerShape->dimensions[1]));
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerfStruct* perfResults = new cudnnConvolutionFwdAlgoPerfStruct[numRequestedConvAlgos];
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(Matrix_GPU::cuda->cudnnHandle,
                                                      forwardInputDesc,
                                                      filtersDesc,
                                                      convDesc,
                                                      forwardOutputDesc,
                                                      numRequestedConvAlgos,
                                                      &returnedAlgoCount,
                                                      perfResults));

    if (returnedAlgoCount != numRequestedConvAlgos)
        throw std::runtime_error("ConvLayer::Compile : Not enough convolution algorithms returned");

    forwardAlgo = perfResults[0].algo;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(Matrix_GPU::cuda->cudnnHandle,
                                                       forwardInputDesc,
                                                       filtersDesc,
                                                       convDesc,
                                                       forwardOutputDesc,
                                                       forwardAlgo,
                                                       &forwardWorkspaceSize));

    if (forwardWorkspaceSize)
    {checkCUDA(cudaMalloc(&forwardWorkspace, forwardWorkspaceSize)); }

    int o_batch, o_channels, o_height, o_width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                     forwardInputDesc,
                                                     filtersDesc,
                                                     &o_batch,
                                                     &o_channels,
                                                     &o_height,
                                                     &o_width));

    if (o_batch != 1)
        throw std::runtime_error("ConvLayer::Compile : Batch size is not 1");
    if (o_channels != filterShape->dimensions[2])
        throw std::runtime_error("ConvLayer::Compile : Output channel count is not equal to filter channel count");
    if (o_height != layerShape->dimensions[0] || o_width != layerShape->dimensions[1])
        throw std::runtime_error("ConvLayer::Compile : Output dimensions are not correct");

    cudnnConvolutionBwdFilterAlgoPerfStruct* b_f_perf_results = new cudnnConvolutionBwdFilterAlgoPerfStruct[numRequestedConvAlgos];
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(Matrix_GPU::cuda->cudnnHandle, forwardInputDesc, forwardOutputDesc,
                                                  convDesc,
                                                  filtersDesc, numRequestedConvAlgos, &returnedAlgoCount,
                                                  b_f_perf_results);
    if (returnedAlgoCount != numRequestedConvAlgos)
        throw std::runtime_error("ConvLayer::Compile : Not enough backward filter algorithms returned");
    backwardFilterAlgo = b_f_perf_results[0].algo;

    cudnnGetConvolutionBackwardFilterWorkspaceSize(Matrix_GPU::cuda->cudnnHandle, forwardInputDesc, forwardOutputDesc,
                                                   convDesc,
                                                   filtersDesc, backwardFilterAlgo, &backwardFilterWorkspaceSize);

    if (backwardFilterWorkspaceSize)
    {checkCUDA(cudaMalloc(&backwardFilterWorkspace, backwardFilterWorkspaceSize)); }

    cudnnConvolutionBwdDataAlgoPerfStruct* b_d_perf_results = new cudnnConvolutionBwdDataAlgoPerfStruct[numRequestedConvAlgos];
    cudnnGetConvolutionBackwardDataAlgorithm_v7(Matrix_GPU::cuda->cudnnHandle, filtersDesc, forwardOutputDesc, convDesc,
                                                forwardInputDesc, numRequestedConvAlgos, &returnedAlgoCount,
                                                b_d_perf_results);
    if (returnedAlgoCount != numRequestedConvAlgos)
        throw std::runtime_error("ConvLayer::Compile : Not enough backward data algorithms returned");
    backwardDataAlgo = b_d_perf_results[0].algo;

    cudnnGetConvolutionBackwardDataWorkspaceSize(Matrix_GPU::cuda->cudnnHandle, filtersDesc, forwardOutputDesc,
                                                 convDesc,
                                                 forwardInputDesc, backwardDataAlgo, &backwardDataWorkspaceSize);
    if (backwardDataWorkspaceSize)
    {checkCUDA(cudaMalloc(&backwardDataWorkspace, backwardDataWorkspaceSize)); }
#else
    rotatedFilter = filters->Copy();
#endif
}

MAT* ConvLayer::FeedForward(const MAT* input)
{
#if USE_GPU
    checkCUDNN(cudnnConvolutionForward(Matrix_GPU::cuda->cudnnHandle,
                                       &Matrix_GPU::cuda->one,
                                       forwardInputDesc,
                                       input->GetData(),
                                       filtersDesc,
                                       filters->GetData(),
                                       convDesc,
                                       forwardAlgo,
                                       forwardWorkspace,
                                       forwardWorkspaceSize,
                                       &Matrix_GPU::cuda->zero,
                                       forwardOutputDesc,
                                       z->GetData()));

    /*checkCUDNN(cudnnAddTensor(Matrix_GPU::cuda->cudnnHandle,
                              &Matrix_GPU::cuda->one,
                              *z->GetDescriptor(),
                              bias->GetData(),
                              &Matrix_GPU::cuda->one,
                              *z->GetDescriptor(),
                              z->GetData()));*/
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

#endif

#if USE_GPU
    activation->FeedForward(z, forwardOutputDesc, result, forwardOutputDesc);
#else
    activation->FeedForward(z, result);
#endif
    return result;

}


//May be optimized by not rotating the matrix
MAT* ConvLayer::BackPropagate(const MAT* lastDelta, const MAT* prevLayerOutput)
{
#if USE_GPU
    /*checkCUDNN(cudnnConvolutionBackwardBias(Matrix_GPU::cuda->cudnnHandle,
                                            &Matrix_GPU::cuda->one,
                                            *lastDelta->GetDescriptor(),
                                            lastDelta->GetData(),
                                            &Matrix_GPU::cuda->one,
                                            *deltaBias->GetDescriptor(),
                                            deltaBias->GetData()));*/

    checkCUDNN(cudnnConvolutionBackwardFilter(Matrix_GPU::cuda->cudnnHandle,
                                              &Matrix_GPU::cuda->one,
                                              forwardInputDesc,
                                              prevLayerOutput->GetData(),
                                              forwardOutputDesc,
                                              lastDelta->GetData(),
                                              convDesc,
                                              backwardFilterAlgo,
                                              backwardFilterWorkspace,
                                              backwardFilterWorkspaceSize,
                                              &Matrix_GPU::cuda->one, // zero ?
                                              filtersDesc,
                                              delta->GetData()));

    checkCUDNN(cudnnConvolutionBackwardData(Matrix_GPU::cuda->cudnnHandle,
                                            &Matrix_GPU::cuda->one,
                                            filtersDesc,
                                            filters->GetData(),
                                            forwardOutputDesc,
                                            lastDelta->GetData(),
                                            convDesc,
                                            backwardDataAlgo,
                                            backwardDataWorkspace,
                                            backwardDataWorkspaceSize,
                                            &Matrix_GPU::cuda->zero,
                                            forwardInputDesc,
                                            nextLayerDelta->GetData()));
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
            Matrix::Convolution(prevLayerOutput, previousDeltaMultiplied, preDelta);
            delta->Add(preDelta, delta);
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
        }
        prevLayerOutput->GoToNextMatrix();
        nextLayerDelta->GoToNextMatrix();
    }

    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    prevLayerOutput->ResetOffset();
    nextLayerDelta->ResetOffset();
#endif
    return nextLayerDelta;
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
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardInputDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardOutputDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filtersDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDA(cudaFree(forwardWorkspace));
    checkCUDA(cudaFree(backwardFilterWorkspace));
    checkCUDA(cudaFree(backwardDataWorkspace));
#else
    delete rotatedFilter;
#endif
}




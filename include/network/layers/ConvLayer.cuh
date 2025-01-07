#pragma once

#include "network/layers/Layer.cuh"
#include "network/LayerShape.cuh"
#include "network/activation/Activation.cuh"
#include "network/Operations.h"
#include "matrix/Matrix.cuh"

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
class ConvLayer
{

public:
    ~ConvLayer();

    LMAT<layerShape>* FeedForward(const LMAT<prevLayerShape>* input);

    LMAT<layerShape>* BackPropagate(const LMAT<layerShape>* delta, const LMAT<prevLayerShape>* prevLayerOutput);

    [[nodiscard]] LMAT<layerShape>* getResult() const;

    void Compile();

    void AddDeltaFrom(Layer<ConvLayer>* ConvLayer);

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    //void SpecificSave(std::ofstream& writer);

    //static Layer<ConvLayer>* Load(std::ifstream& reader);

    std::string getLayerTitle();

    Layer<ConvLayer>* Clone();

private:

#if not USE_GPU
    void FlipAndCenterFilter();
    void GetOperationsForFullConvolution();
#endif

    LMAT<layerShape>* result = nullptr;
    MAT* filters = nullptr;
    //Delta for next layer
    LMAT<layerShape>* delta = nullptr;
    MAT* preDelta = nullptr;
    MAT* activationDelta;
    MAT* z;
    MAT* previousDeltaMultiplied;
    MAT<1,1,dimCount>* bias;
    MAT<1,1,dimCount>* deltaBias;

    MAT* nextLayerDelta = nullptr;
    MAT* nextLayerDeltaTemp = nullptr;

#if USE_GPU
    cudnnFilterDescriptor_t filtersDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t forwardAlgo;
    size_t forwardWorkspaceSize = 0;
    float* forwardWorkspace = nullptr;
    cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t backwardDataAlgo;
    size_t backwardFilterWorkspaceSize = 0;
    size_t backwardDataWorkspaceSize = 0;
    float* backwardFilterWorkspace = nullptr;
    float* backwardDataWorkspace = nullptr;

    static inline const int numRequestedConvAlgos = 1;

    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc, biasDesc;
#else
    MAT* rotatedFilter = nullptr;
#endif

    //Result from the previous layer (don't initialize when compiling the layer)
    uint filterCount = filterShape::z;
    uint preivousDimCount = prevLayerShape::z;
    uint dimCount = filterCount * preivousDimCount;
    uint offset = 0;

    std::vector<Operation*> FullConvOperations = std::vector<Operation*>();

    Optimizer* optimizer = nullptr;
};

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::Compile()
{
    //Check if the previous layer has 3 dimensions, if not throw an error
    if (prevLayerShape::z < 3)
    {
        throw std::invalid_argument("Input of a CNN network must have 3 dimensions");
    }

    //If there is no activation function, throw an error
    if (activation == nullptr)
    {
        throw std::invalid_argument("ConvLayer : Must have an activation function !");
    }

    //const int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    //const int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;

    //Number of filter per channel
    //filterCount = filterShape::z;
    //Number of channel in the previous layer
    //preivousDimCount = prevLayerShape::z;

    //Number of dimCount
    //dimCount = filterCount * preivousDimCount;

    //If the filters has no been initialized, create it and initialize it with random values
    if (filters == nullptr)
    {
        filters = new MAT(filterShape::z, filterShape::y, (int) dimCount);
        //Function to init the filters with random values
        WeightsInit::HeUniform(filterShape::z * filterShape::y, filters);
    }

    nextLayerDelta = new MAT(prevLayerShape::z, prevLayerShape::y,
                             prevLayerShape::z);

    nextLayerDeltaTemp = new MAT(prevLayerShape::z, prevLayerShape::y);


    delta = filters->Copy();
    delta->Zero();
    preDelta = new MAT(filters->GetRows(), filters->GetCols());


//    layerShape = new LayerShape(previousLayer::z - filters->GetRows() + 1, previousLayer::y -
//                                                                                       filters->GetCols() + 1,
//                                dimCount);

    result = new MAT(layerShape::z, layerShape::y, layerShape::z);

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    offset = previousDeltaMultiplied->GetRows() - 1;
    activationDelta = result->Copy();

    bias = new MAT<1, 1, (int) dimCount>();
#if USE_GPU
    float* biasValues = new float[bias->GetSize()];
    for (int i = 0; i < bias->GetSize(); i++)
        biasValues[i] = 0.01;

    checkCUDA(cudaMemcpy(bias->GetData(), biasValues, bias->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
    delete[] biasValues;
#else
    for (int i = 0; i < bias->GetSize(); i++)
    {
        (*bias)[i] = 0.01;
    }
#endif
    deltaBias = new MAT<1, 1, (int) dimCount>();

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

    checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          filterShape->dimensions[2],
                                          1,
                                          1));

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

#endif
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
LMAT<layerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape>::FeedForward(const LMAT<prevLayerShape>* input)
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

    checkCUDNN(cudnnAddTensor(Matrix_GPU::cuda->cudnnHandle,
                              &Matrix_GPU::cuda->one,
                              biasDesc,
                              bias->GetData(),
                              &Matrix_GPU::cuda->one,
                              forwardOutputDesc,
                              z->GetData()))
#else
    //Reshape the layer in case it does not have the right shape - cannot happen with templates
    // result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
    //result->PrintSize();
    //Loop through all the dimensions of the previous layer
    for (uint j = 0; j < preivousDimCount; j++)
    {
        //Loop through all the dimensions of the actual layer
        for (int i = 0; i < filterCount; i++)
        {
            //Apply convolution between input and filters and output it in z
            LMAT<layerShape>::Convolution<FILTER_SIZE, STRIDE>(input, filters, z);

            //Add the bias to the result
            for (int k = 0; k < z->GetRows() * z->GetCols(); k++)
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

#endif

    //Apply activation function on all the matrix
#if USE_GPU
    activation->FeedForward(z, forwardOutputDesc, result, forwardOutputDesc);
#else
    activation->FeedForward(z, result);
#endif

    return result;
}

#if not USE_GPU

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::FlipAndCenterFilter()
{
    for (int d = 0; d < filters->GetDims(); d++)
    {
        for (int i = 0; i < filters->GetCols(); ++i)
        {
            for (int j = 0; j < filters->GetRows(); ++j)
            {
                (*rotatedFilter)(i + offset, j + offset) = (*filters)(filters->GetRows() - 1 - j,
                                                                      filters->GetCols() - 1 - i);
            }
        }
        rotatedFilter->GoToNextMatrix();
        filters->GoToNextMatrix();
    }

    rotatedFilter->ResetOffset();
    filters->ResetOffset();

}

#endif

//May be optimized by not rotating the matrix
template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
LMAT<layerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape>::BackPropagate(const LMAT<layerShape>* delta, const LMAT<prevLayerShape>* prevLayerOutput)
{
    //Set to zero the delta of the next layer
    nextLayerDelta->Zero();
#if USE_GPU
    checkCUDNN(cudnnConvolutionBackwardBias(Matrix_GPU::cuda->cudnnHandle,
                                            &Matrix_GPU::cuda->one,
                                            forwardOutputDesc,
                                            lastDelta->GetData(),
                                            &Matrix_GPU::cuda->one,
                                            biasDesc,
                                            deltaBias->GetData()));

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
            Matrix::Flip180(filters, rotatedFilter);

            //Calculate the partial derivative for the previous layer
            //Matrix::FullConvolution(rotatedFilter,previousDeltaMultiplied,nextLayerDeltaTemp);

            //Accumulate the result of the partial derivative
            //nextLayerDelta->Add(nextLayerDeltaTemp,nextLayerDelta);



            //Calculate the partial derivative of the weights
            Matrix::Convolution(prevLayerOutput, previousDeltaMultiplied, preDelta);

            //Accumulate the result
            delta->Add(preDelta, delta);

            //Filters, rotatedFilter, previousDeltaMultiplied and delta are moved to the next matrix
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            previousDeltaMultiplied->GoToNextMatrix();
            delta->GoToNextMatrix();
        }
        // Input and nextLayerDelta are moved to the next matrix
        prevLayerOutput->GoToNextMatrix();
        nextLayerDelta->GoToNextMatrix();
    }
    //Resetting all the matrix offset
    nextLayerDelta->ResetOffset();
    delta->ResetOffset();
    filters->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    prevLayerOutput->ResetOffset();
#endif

    //Return the partial derivative for the previous layer
    return nextLayerDelta;
}


template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(delta, filters);

#if USE_GPU
    //ToDo: Make this run on GPU
    Matrix deltaBiasCPU(delta->GetRows(), deltaBias->GetCols(), deltaBias->GetDims(), deltaBias->GetData_CPU());
    Matrix deltaCPU(delta->GetRows(), delta->GetCols(), delta->GetDims(), delta->GetData_CPU());
    for (int i = 0; i < deltaBias->GetDims(); i++)
    {
        for (int j = 0; j < delta->GetRows() * delta->GetCols(); j++)
        {
            deltaBiasCPU[0] += deltaCPU[j];
        }
        deltaBiasCPU.GoToNextMatrix();
        deltaCPU.GoToNextMatrix();
    }

    deltaBiasCPU.ResetOffset();
    deltaCPU.ResetOffset();

    checkCUDA(cudaMemcpy(deltaBias->GetData(), deltaBiasCPU.GetData(), deltaBias->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
#else
    for (int i = 0; i < deltaBias->GetDims(); i++)
    {
        for (int j = 0; j < delta->GetRows() * delta->GetCols(); j++)
        {
            (*deltaBias)[0] += (*delta)[j];
        }
        deltaBias->GoToNextMatrix();
        delta->GoToNextMatrix();
    }

    deltaBias->ResetOffset();
    delta->ResetOffset();
#endif

    optimizer->Compute(deltaBias, bias, bias->GetSize());
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
LMAT<layerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape>::getResult() const
{
    return result;
}


template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::AddDeltaFrom(Layer<ConvLayer<activation, prevLayerShape, layerShape, filterShape>>* Layer)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::AddDeltaFrom is not implemented on GPU");
#else
    auto* convLayer = (ConvLayer*) Layer;

    delta->AddAllDims(convLayer->delta, delta);
    deltaBias->AddAllDims(convLayer->deltaBias, deltaBias);
#endif
}

//template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
//Layer<ConvLayer<activation, prevLayerShape, layerShape, filterShape>>* ConvLayer<activation, prevLayerShape, layerShape, filterShape>::Load(std::ifstream& reader)
//{
//#if USE_GPU
//    throw std::runtime_error("ConvLayer::Load is not implmentedd on GPU");
//#else
//    Matrix* filters = Matrix::Read(reader);
//    LayerShape* filterShape = LayerShape::Load(reader);
//    Activation<Args>* activation = Activation<Args>::Read(reader);
//    return new ConvLayer(filters, filterShape, activation);
//#endif
//}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::ClearDelta()
{
    delta->Zero();
    deltaBias->Zero();
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
std::string ConvLayer<activation, prevLayerShape, layerShape, filterShape>::getLayerTitle()
{
    std::string buf;
    buf += "Convolutional layer\n";
    buf += "Filter count per channel : " + std::to_string(filterShape::z) + "\n";
    buf += "Feature map count : " + std::to_string(layerShape::z) + "\n";
    buf += "Output size : " + layerShape::GetDimensions() + "\n";
    return buf;
}


//template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
//void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::SpecificSave(std::ofstream& writer)
//{
//    filters->Save(writer);
//    filterShape->Save(writer);
//    activation->Save(writer);
//}

#if not USE_GPU

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::GetOperationsForFullConvolution()
{
    const int outputCols = previousDeltaMultiplied->GetCols() + filters->GetCols() - 1;
    const int outputRows = previousDeltaMultiplied->GetRows() + filters->GetRows() - 1;

    const int filterCols = filters->GetCols();
    const int filterRows = filters->GetRows();

    const int inputCols = previousDeltaMultiplied->GetCols();
    const int inputRows = previousDeltaMultiplied->GetRows();



    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            for (int k = 0; k < filterCols; k++)
            {
                for (int l = 0; l < filterRows; l++)
                {
                    const int inputRow = i - k;
                    const int inputCol = j - l;
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                    {
                        float* filterPointer = &((*rotatedFilter)(k, l));
                        float* matrixPointer = &((*previousDeltaMultiplied)(inputRow, inputCol));
                        FullConvOperations.push_back(
                                new MulAddTo1(filterPointer, matrixPointer, &((*nextLayerDelta)(i, j)), 1));
                    }
                }

            }
        }
    }
}

#endif

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
Layer<ConvLayer<activation, prevLayerShape, layerShape, filterShape>>* ConvLayer<activation, prevLayerShape, layerShape, filterShape>::Clone()
{
    auto* filterCopy = filters->CopyWithSameData();
    return new ConvLayer(filterCopy, new LayerShape(filterShape::z, filterShape::y,
                                                    filterShape::z), activation);
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape>::AverageGradients(const int batchSize)
{
    delta->DivideAllDims(batchSize);
    deltaBias->DivideAllDims(batchSize);
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape>
ConvLayer<activation, prevLayerShape, layerShape, filterShape>::~ConvLayer()
{
    delete filters;
    //delete filterShape;
    //delete activation;
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
    checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filtersDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDA(cudaFree(forwardWorkspace));
    checkCUDA(cudaFree(backwardFilterWorkspace));
    checkCUDA(cudaFree(backwardDataWorkspace));
#else
    delete rotatedFilter;
#endif
}




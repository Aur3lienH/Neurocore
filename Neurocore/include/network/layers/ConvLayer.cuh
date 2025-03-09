#pragma once

#include "network/layers/Layer.cuh"
#include "network/LayerShape.cuh"
#include "network/activation/Activation.cuh"
#include "network/Operations.h"
#include "matrix/Matrix.cuh"

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer = Optimizer<Constant<0.001>>, bool GPU = GPU_DEFAULT, bool test = false>
class ConvLayer final
{

public:

    using KernelShape = filterShape;

    using Shape = layerShape;

    ~ConvLayer();

    LMAT<layerShape>* FeedForward(const LMAT<prevLayerShape>* input);

    LMAT<prevLayerShape>* BackPropagate(const LMAT<layerShape>* delta, const LMAT<prevLayerShape>* prevLayerOutput);

    [[nodiscard]] LMAT<layerShape>* getResult() const;

    void Compile();

    void AddDeltaFrom(Layer<ConvLayer>* ConvLayer);

    void AverageGradients(int batchSize);

    void ClearDelta();

    void UpdateWeights(double learningRate, int batchSize);

    void Save(std::ofstream& writer);

    void Load(std::ifstream& reader);

    //static Layer<ConvLayer>* Load(std::ifstream& reader);

    std::string getLayerTitle();

    Layer<ConvLayer>* Clone();

    LMAT<filterShape> getZ() requires(test);

    static const uint filterCount = filterShape::z;
    static const uint preivousDimCount = prevLayerShape::z;
    static const uint dimCount = filterCount * preivousDimCount;

    void SetWeights(MAT<filterShape::x, filterShape::y, dimCount>* weights) requires(test);

    void SetBiases(MAT<1,1,filterShape::z>* biases) requires(test);


    // !!! Only for tests  !!!

    const LMAT<filterShape>* getFilters() const requires(test)
    {
        return filters;
    }

    const LMAT<layerShape>* getBias() const requires(test)
    {
        return bias;
    }

    const MAT<filterShape::x, filterShape::y, dimCount>* getDelta() const requires(test)
    {
        return delta;
    }

    const MAT<1,1,filterShape::z>* getDeltaBias() const requires(test)
    {
        return deltaBias;
    }



private:

    void FlipAndCenterFilter() requires (!GPU);
    void GetOperationsForFullConvolution()  requires (!GPU);
    //Result from the previous layer (don't initialize when compiling the layer)
    //Result from the previous layer (don't initialize when compiling the layer)
    uint offset = 0;

    std::vector<Operation*> FullConvOperations = std::vector<Operation*>();

    //Optimizer* optimizer = nullptr;

    LMAT<layerShape>* result = nullptr;
    MAT<filterShape::x, filterShape::y, dimCount>* filters = nullptr;
    //Delta for next layer
    MAT<filterShape::x, filterShape::y, dimCount>* delta = nullptr;
    MAT<filterShape::x, filterShape::y, dimCount>* preDelta = nullptr;
    LMAT<layerShape>*activationDelta = nullptr;
    LMAT<layerShape>*z = nullptr;
    MAT<layerShape::x,layerShape::y,1>* tempZ = nullptr;
    LMAT<layerShape>*previousDeltaMultiplied = nullptr;
    MAT<1,1,filterShape::z>*bias = nullptr;
    MAT<1,1,filterShape::z>*deltaBias = nullptr;

    LMAT<prevLayerShape>* nextLayerDelta = nullptr;
    LMAT<prevLayerShape>* nextLayerDeltaTemp = nullptr;

    MAT<filterShape::x, filterShape::y, dimCount>* rotatedFilter = nullptr;

    // GPU stuff
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
};

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::SetWeights(MAT<filterShape::x, filterShape::y, dimCount>* weights) requires(test)
{
    delete filters;
    filters = weights->Copy();
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::SetBiases(MAT<1,1,filterShape::z>* biases) requires(test)
{
    delete bias;
    bias = biases->Copy();
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::Compile()
{
    static_assert(prevLayerShape::x && prevLayerShape::y, "Input of a CNN network must have 3 dimensions");

    //const int outputRow = previousLayer->dimensions[0] - filterShape->dimensions[0] + 1;
    //const int outputCol = previousLayer->dimensions[1] - filterShape->dimensions[1] + 1;

    //Number of filter per channel
    //filterCount = filterShape::z;
    //Number of channel in the previous layer
    //preivousDimCount = prevLayerShape::z;

    //Number of dimCount
    //dimCount = filterCount * preivousDimCount;

    //If the filters has no been initialized, create it and initialize it with random values
    std::cout << filters << " this is filters ! \n";
    if (filters == nullptr)
    {
        filters = new MAT<filterShape::x, filterShape::y, dimCount>();
        //Function to init the filters with random values
        WeightsInit::HeUniform(prevLayerShape::x * prevLayerShape::y * prevLayerShape::z, filters);
    }

    nextLayerDelta = new LMAT<prevLayerShape>();

    nextLayerDeltaTemp = new LMAT<prevLayerShape>();


    delta = filters->Copy();
    delta->Zero();
    preDelta = filters->Copy();
    preDelta->Zero();
    tempZ = new MAT<layerShape::x, layerShape::y, 1>();


    //    layerShape = new LayerShape(previousLayer::x - filters::GetRows() + 1, previousLayer::y -
    //                                                                                       filters::GetCols() + 1,
    //                                dimCount);

    result = new LMAT<layerShape>();

    z = result->Copy();

    previousDeltaMultiplied = result->Copy();
    offset = layerShape::x - 1; // previousDeltaMultiplied.rows -
    activationDelta = result->Copy();

    bias = new MAT<1, 1, filterShape::z>();
    if constexpr (GPU)
    {
        float* biasValues = new float[bias->GetSize()];
        for (int i = 0; i < bias->GetSize(); i++)
            biasValues[i] = 0.01;

        checkCUDA(cudaMemcpy(bias->GetData(), biasValues, bias->GetSize() * sizeof(float), cudaMemcpyHostToDevice));
        delete[] biasValues;
    }
    else
    {
        for (int i = 0; i < bias->GetSize(); i++)
        {
            bias->data[i] = 0.01;
        }
    }
    deltaBias = new MAT<1, 1, filterShape::z>();

    optimizer::Compile(filters->GetSize() + bias->GetSize());

    if constexpr (GPU)
    {
        checkCUDNN(cudnnCreateFilterDescriptor(&filtersDesc));
        checkCUDNN(cudnnSetFilter4dDescriptor(filtersDesc,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            filterShape::z,
            1,
            filterShape::x,
            filterShape::y));

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
            prevLayerShape::x,
            prevLayerShape::y));

        checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDesc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            filterShape::z,
            layerShape::x,
            layerShape::y));

        checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
        checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            filterShape::z,
            1,
            1));

        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerfStruct* perfResults = new cudnnConvolutionFwdAlgoPerfStruct[numRequestedConvAlgos];
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cuda->cudnnHandle,
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

        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cuda->cudnnHandle,
            forwardInputDesc,
            filtersDesc,
            convDesc,
            forwardOutputDesc,
            forwardAlgo,
            &forwardWorkspaceSize));

        if (forwardWorkspaceSize)
        {
            checkCUDA(cudaMalloc(&forwardWorkspace, forwardWorkspaceSize));
        }

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
        if (o_channels != filterShape::z)
            throw std::runtime_error("ConvLayer::Compile : Output channel count is not equal to filter channel count");
        if (o_height != layerShape::x || o_width != layerShape::y)
            throw std::runtime_error("ConvLayer::Compile : Output dimensions are not correct");

        cudnnConvolutionBwdFilterAlgoPerfStruct* b_f_perf_results = new cudnnConvolutionBwdFilterAlgoPerfStruct[
            numRequestedConvAlgos];
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda->cudnnHandle, forwardInputDesc, forwardOutputDesc,
                                                      convDesc,
                                                      filtersDesc, numRequestedConvAlgos, &returnedAlgoCount,
                                                      b_f_perf_results);
        if (returnedAlgoCount != numRequestedConvAlgos)
            throw std::runtime_error("ConvLayer::Compile : Not enough backward filter algorithms returned");
        backwardFilterAlgo = b_f_perf_results[0].algo;

        cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda->cudnnHandle, forwardInputDesc, forwardOutputDesc,
                                                       convDesc,
                                                       filtersDesc, backwardFilterAlgo, &backwardFilterWorkspaceSize);

        if (backwardFilterWorkspaceSize)
        {
            checkCUDA(cudaMalloc(&backwardFilterWorkspace, backwardFilterWorkspaceSize));
        }

        cudnnConvolutionBwdDataAlgoPerfStruct* b_d_perf_results = new cudnnConvolutionBwdDataAlgoPerfStruct[
            numRequestedConvAlgos];
        cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda->cudnnHandle, filtersDesc, forwardOutputDesc, convDesc,
                                                    forwardInputDesc, numRequestedConvAlgos, &returnedAlgoCount,
                                                    b_d_perf_results);
        if (returnedAlgoCount != numRequestedConvAlgos)
            throw std::runtime_error("ConvLayer::Compile : Not enough backward data algorithms returned");
        backwardDataAlgo = b_d_perf_results[0].algo;

        cudnnGetConvolutionBackwardDataWorkspaceSize(cuda->cudnnHandle, filtersDesc, forwardOutputDesc,
                                                     convDesc,
                                                     forwardInputDesc, backwardDataAlgo, &backwardDataWorkspaceSize);
        if (backwardDataWorkspaceSize)
        {
            checkCUDA(cudaMalloc(&backwardDataWorkspace, backwardDataWorkspaceSize));
        }
    }

    else
    {
        rotatedFilter = filters->Copy();
        for (int j = 0; j < preivousDimCount; j++)
        {
            for (int i = 0; i < filterCount; i++)
            {
                //GetOperationsForFullConvolution();
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
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
LMAT<layerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::FeedForward(const LMAT<prevLayerShape>* input)
{
    if constexpr (GPU)
    {
        checkCUDNN(cudnnConvolutionForward(cuda->cudnnHandle,
            &cuda->one,
            forwardInputDesc,
            input->GetData(),
            filtersDesc,
            filters->GetData(),
            convDesc,
            forwardAlgo,
            forwardWorkspace,
            forwardWorkspaceSize,
            &cuda->zero,
            forwardOutputDesc,
            z->GetData()));

        checkCUDNN(cudnnAddTensor(cuda->cudnnHandle,
            &cuda->one,
            biasDesc,
            bias->GetData(),
            &cuda->one,
            forwardOutputDesc,
            z->GetData()))
    }
    else
    {
        //Reshape the layer in case it does not have the right shape - cannot happen with templates
        // result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
        //result->PrintSize();
        //Loop through all the dimensions of the previous layer
        z->Zero();
        for (uint j = 0; j < filterShape::z; j++)
        {
            //Loop through all the dimensions of the actual layer
            for (int i = 0; i < preivousDimCount; i++)
            {
                //Apply convolution between input and filters and output it in z
                LMAT<prevLayerShape>::template Convolution<filterShape::x, 1>(input, filters, tempZ);
                z->Add(tempZ, z);


                //Filters and bias are moved to the next matrix
                filters->GoToNextMatrix();
                input->GoToNextMatrix();
            }
            //Add the bias to the result
            for (int k = 0; k < layerShape::x * layerShape::y; k++)
            {
                (*z).set(k, bias[0].get(0) + (*z).get(k));
            }
            //Input is moved to the next matrix
            bias->GoToNextMatrix();
            z->GoToNextMatrix();
            input->ResetOffset();
        }
        //All the matrix offset are reset
        filters->ResetOffset();
        input->ResetOffset();
        bias->ResetOffset();
        z->ResetOffset();
    }

    //Apply activation function on all the matrix
    activation::FeedForward(z, result);

    return result;
}

template <typename activation, typename prevLayerShape, typename layerShape, typename filterShape, typename optimizer,
          bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::FlipAndCenterFilter() requires (!
    GPU)
{
    for (int d = 0; d < filters->GetDims(); d++)
    {
        for (int i = 0; i < filterShape::y; ++i)
        {
            for (int j = 0; j < filterShape::x; ++j)
            {
                (*rotatedFilter)(i + offset, j + offset) = (*filters)(filterShape::x - 1 - j,
                                                                      filterShape::y - 1 - i);
            }
        }
        rotatedFilter->GoToNextMatrix();
        filters->GoToNextMatrix();
    }

    rotatedFilter->ResetOffset();
    filters->ResetOffset();
}

//May be optimized by not rotating the matrix
template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
LMAT<prevLayerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::BackPropagate(const LMAT<layerShape>* lastDelta, const LMAT<prevLayerShape>* prevLayerOutput)
{
    //Set to zero the delta of the next layer
    nextLayerDelta->Zero();//Calculate the partial derivative of the activation function
    activation::Derivative(z, activationDelta, result, lastDelta);
    if constexpr (GPU)
    {
        // Todo: check if we shouldn't be using activationDelta instead  of lastDelta
        checkCUDNN(cudnnConvolutionBackwardBias(cuda->cudnnHandle,
            &cuda->one,
            forwardOutputDesc,
            lastDelta->GetData(),
            &cuda->one,
            biasDesc,
            deltaBias->GetData()));

        checkCUDNN(cudnnConvolutionBackwardFilter(cuda->cudnnHandle,
            &cuda->one,
            forwardInputDesc,
            prevLayerOutput->GetData(),
            forwardOutputDesc,
            lastDelta->GetData(),
            convDesc,
            backwardFilterAlgo,
            backwardFilterWorkspace,
            backwardFilterWorkspaceSize,
            &cuda->one, // zero ?
            filtersDesc,
            delta->GetData()));

        checkCUDNN(cudnnConvolutionBackwardData(cuda->cudnnHandle,
            &cuda->one,
            filtersDesc,
            filters->GetData(),
            forwardOutputDesc,
            lastDelta->GetData(),
            convDesc,
            backwardDataAlgo,
            backwardDataWorkspace,
            backwardDataWorkspaceSize,
            &cuda->zero,
            forwardInputDesc,
            nextLayerDelta->GetData()));
    }
    else
    {
        //Multiply the partial derivative of the activation function with the partial derivative of the previous layer
    lastDelta->MultiplyAllDims(activationDelta, previousDeltaMultiplied);


    //Loop through all the dimensions of the previous layer
    for (uint i = 0; i < filterCount; i++)
    {
        //Loop through all the dimensions of the actual layer
        for (uint j = 0; j < preivousDimCount; j++)
        {
            //Flip the filter
            MAT<filterShape::x, filterShape::y, dimCount>::Flip180(filters, rotatedFilter);

            //Calculate the partial derivative for the previous layer
            LMAT<filterShape>::FullConvolution(rotatedFilter,previousDeltaMultiplied,nextLayerDeltaTemp);

            //Accumulate the result of the partial derivative
            nextLayerDelta->Add(nextLayerDeltaTemp,nextLayerDelta);


            //Calculate the partial derivative of the weights
            LMAT<prevLayerShape>::template Convolution<layerShape::x, 1>(prevLayerOutput, previousDeltaMultiplied, preDelta);

            //Accumulate the result
            delta->Add(preDelta, delta);


            //Filters, rotatedFilter, previousDeltaMultiplied and delta are moved to the next matrix
            filters->GoToNextMatrix();
            rotatedFilter->GoToNextMatrix();
            delta->GoToNextMatrix();
            prevLayerOutput->GoToNextMatrix();
        }
        // Input and nextLayerDelta are moved to the next matrix
        previousDeltaMultiplied->GoToNextMatrix();
        prevLayerOutput->ResetOffset();
    }
    //Resetting all the matrix offset
    filters->ResetOffset();
    nextLayerDelta->ResetOffset();
    delta->ResetOffset();
    rotatedFilter->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    prevLayerOutput->ResetOffset();

    for (int i = 0; i < filterCount; i++)
    {
        for (int j = 0; j < previousDeltaMultiplied->GetMatrixSize(); j++)
        {
            (*deltaBias).set(0, (*deltaBias).get(0) + (*previousDeltaMultiplied).get(j));
        }
        deltaBias->GoToNextMatrix();
        previousDeltaMultiplied->GoToNextMatrix();
    }

    deltaBias->ResetOffset();
    previousDeltaMultiplied->ResetOffset();
    //throw std::runtime_error("Do not work over there ! ");
    }

    //Return the partial derivative for the previous layer
    return nextLayerDelta;
}


template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer::Compute(delta, filters);

    if constexpr (GPU) //ToDo: Make this run on GPU
    {
        Matrix deltaBiasCPU(deltaBias->GetData_CPU());
        Matrix deltaCPU(delta->GetData_CPU());
        for (int i = 0; i < deltaBias->GetDims(); i++)
        {
            for (int j = 0; j < delta->GetRows() * delta->GetCols(); j++)
            {
                deltaBiasCPU.set(0, deltaBiasCPU.get(0) + deltaCPU.get(j));
            }
            deltaBiasCPU.GoToNextMatrix();
            deltaCPU.GoToNextMatrix();
        }

        deltaBiasCPU.ResetOffset();
        deltaCPU.ResetOffset();

        checkCUDA(cudaMemcpy(deltaBias->GetData(), deltaBiasCPU.GetData(), deltaBias->GetSize() * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    optimizer::Compute(deltaBias, bias, bias->GetSize());
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
LMAT<layerShape>* ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::getResult() const
{
    return result;
}


template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::AddDeltaFrom(Layer<ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>>* Layer)
{
    if constexpr (GPU)
        {throw std::runtime_error("ConvLayer::AddDeltaFrom is not implemented on GPU");}
    else
    {
        auto* convLayer = (ConvLayer*) Layer;

        delta->AddAllDims(convLayer->delta, delta);
        deltaBias->AddAllDims(convLayer->deltaBias, deltaBias);
    }
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::Load(std::ifstream &reader)
{
#if USE_GPU
    throw std::runtime_error("ConvLayer::Load is not implmentedd on GPU");
#else
    this->filters = LMAT<filterShape>::Read(reader);
    this->bias = MAT<1,1,layerShape::z>::Read(reader);
#endif
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::ClearDelta()
{
    delta->Zero();
    deltaBias->Zero();
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
std::string ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::getLayerTitle()
{
    std::string buf;
    buf += "Layer : Convolutional layer\n";
    buf += "Filter count per channel : " + std::to_string(filterShape::z) + "\n";
    buf += "Feature map count : " + std::to_string(layerShape::z) + "\n";
    buf += "Output size : " + layerShape::GetDimensions() + "\n";
    return buf;
}


template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::Save(std::ofstream& writer)
{
    filters->Save(writer);
    bias->Save(writer);
}

/*
template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::GetOperationsForFullConvolution()
{
    const int outputCols = layerShape::y + filterShape::y - 1;
    const int outputRows = layerShape::x + filterShape::x - 1;

    const int filterCols = filterShape::y;
    const int filterRows = filterShape::x;

    const int inputCols = layerShape::y;
    const int inputRows = layerShape::x;



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
                        float* filterPointer = rotatedFilter->data + k * rotatedFilter->GetCols() + l;
                        float* matrixPointer = previousDeltaMultiplied->data + inputRow * previousDeltaMultiplied->GetCols() + inputCol;
                        FullConvOperations.push_back(
                                new MulAddTo1(filterPointer, matrixPointer, nextLayerDelta->data + i * nextLayerDelta->GetCols() + j, 1));
                    }
                }

            }
        }
    }
}

*/

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
Layer<ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>>* ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::Clone()
{
    auto* filterCopy = filters->CopyWithSameData();
    auto cl = new ConvLayer<activation, prevLayerShape, layerShape,  filterShape, optimizer>();
    cl->filters = filterCopy;

    return cl;
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
void ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::AverageGradients(const int batchSize)
{
    delta->DivideAllDims(batchSize);
    deltaBias->DivideAllDims(batchSize);
}

template<typename activation,typename prevLayerShape,typename layerShape, typename filterShape, typename optimizer, bool GPU, bool test>
ConvLayer<activation, prevLayerShape, layerShape, filterShape, optimizer, GPU, test>::~ConvLayer()
{
    delete filters;
    //delete filterShape;
    //delete activation;
    delete result;
    delete z;
    delete delta;
    delete deltaBias;
    delete preDelta;
    delete previousDeltaMultiplied;
    delete activationDelta;
    delete nextLayerDelta;
    delete tempZ;
    delete bias;
    delete nextLayerDeltaTemp;

    if constexpr (GPU)
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(forwardInputDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(forwardOutputDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filtersDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDA(cudaFree(forwardWorkspace));
        checkCUDA(cudaFree(backwardFilterWorkspace));
        checkCUDA(cudaFree(backwardDataWorkspace));
    }
    else
    {delete rotatedFilter;}
}

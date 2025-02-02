#pragma once

#include "matrix/Matrix.cuh"
#include "network/activation/Activation.cuh"
#include "network/layers/Layer.cuh"
#include "tools/Serializer.h"
#include "network/LayerShape.cuh"
#include "network/optimizers/Constant.h"

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer = Constant<0.001>, bool test = false>
class FCL final
{
public:
    FCL();

    ~FCL();

    using Shape = layershape;

    //Cannot do that since templated
    //FCL(MAT* weights, MAT* bias, MAT* delta,
    //    MAT* deltaActivation);

    //Compute the input threw the layer





    MAT<layershape::x>* FeedForward(const MAT<prevLayerShape::x>*);

    //Compute partial derivative (named delta)
    const MAT<prevLayerShape::x>* BackPropagate(const MAT<layershape::x>* delta, const MAT<prevLayerShape::x>* lastWeights);

    //Getter for delta
    const MAT<layershape::x, prevLayerShape::x>* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const MAT<layershape::x>* getDeltaBiases();

    //Getter for the result of the layer
    [[nodiscard]] const MAT<layershape::x>* getResult() const;

    //Clear partial derivative (named delta)
    void ClearDelta();

    //Update the current weights thanks to partial derivative (named delta)
    void UpdateWeights(double learningRate, int batchSize);

    //Add Delta from another identical layer
    void AddDeltaFrom(Layer<FCL<activation,prevLayerShape,layershape,optimizer,test>>* otherLayer);

    //Initialize variable and check for network architecture
    void Compile();

    //Return information on the layer (neurons count)
    std::string getLayerTitle();

    //Clone layer
    Layer<FCL>* Clone();


    void SetWeights(MAT<layershape::x, prevLayerShape::x>* weights) requires(test)
    {
        if (Weights != nullptr)
            delete Weights;
        Weights = weights;
    }

    void SetBiases(MAT<layershape::x>* biases) requires(test)
    {
        if (Biases != nullptr)
            delete Biases;
        Biases = biases;
    }

    Matrix<layershape::x, prevLayerShape::x, 1>* GetWeights() requires(test)
    {
        return Weights;
    }

    Matrix<layershape::x>* GetBiases() requires(test)
    {
        return Biases;
    }


    //!!! Disable because of the templates
    //static FCL* Load(std::ifstream& ifstream);

    //void SpecificSave(std::ofstream& filename);

    void AverageGradients(int batchSize);

#if USE_GPU

    void Save(const std::string& folderPath, int n);

#else
    void Compare(const std::string& folderPath, int n);
#endif

protected:
//Partial derivative of the weights
    MAT<layershape::x,prevLayerShape::x>* Delta = nullptr;

    //Partial derivative of the biases
    MAT<layershape::x>* DeltaBiases = nullptr;

    //Results of the layer
    MAT<layershape::x>* Result = nullptr;

    MAT<layershape::x, prevLayerShape::x>* Weights = nullptr;
    MAT<layershape::x>* Biases = nullptr;

    //Result before passing through the activation function
    MAT<layershape::x>* z = nullptr;

private:
#if USE_GPU
    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
#else
    const MAT<layershape::x ,prevLayerShape::x>* BackPropagateSSE2(const MAT<layershape::x>* delta, const MAT<layershape::x>* lastWeights);

    const MAT<layershape::x ,prevLayerShape::x>* BackPropagateAX2(const MAT<layershape::x>* delta, const MAT<layershape::x>* lastWeights);
#endif

    //Delta passed to the previous network in backpropagation
    MAT<prevLayerShape::x>* newDelta = nullptr;

    //Delta from the activation function
    MAT<layershape::x>* deltaActivation = nullptr;
    //Neurons in the previous layer
    int previousNeuronsCount;

    float* buffer;


};



#include "network/layers/FCL.cuh"
#include <iostream>
#include <fstream>
#include "matrix/Matrix.cuh"
#include "network/LayerShape.cuh"

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
FCL<activation,prevLayerShape,layershape,optimizer,test>::FCL()
{

}

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::Compile()
{
    buffer = new float[8];

    if (Weights == nullptr)
    {
        Weights = activation::InitWeights();
    }
    if (Delta == nullptr)
        Delta = new MAT<layershape::x, prevLayerShape::x>();
    if (deltaActivation == nullptr)
        deltaActivation = new MAT<layershape::x>();
    if (DeltaBiases == nullptr)
        DeltaBiases = new MAT<layershape::x>();
    if (Biases == nullptr)
        Biases = activation::InitBiases();
    if (Result == nullptr)
        Result = new MAT<layershape::x>();
    z = new MAT<layershape::x >();
    newDelta = new MAT<prevLayerShape::x>();

    optimizer::Compile(layershape::x * previousNeuronsCount + layershape::x);

#if USE_GPU
    checkCUDNN(cudnnCreateTensorDescriptor(&forwardInputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardInputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          previousNeuronsCount,
                                          1));
    checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDesc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          1,
                                          NeuronsCount,
                                          1));
#endif
}

#if USE_GPU

FCL::FCL(const int NeuronsCount, Activation* _activation, Matrix_GPU* weights, Matrix_GPU* bias, Matrix_GPU* delta,
         Matrix_GPU* deltaBiases)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    this->NeuronsCount = NeuronsCount;
    this->activation = _activation;
    Weights = weights;
    Biases = bias;
}

Matrix_GPU* FCL::FeedForward(const Matrix_GPU* input)
{
    input->Flatten();
    Matrix_GPU::Multiply(*Weights, *input, *Result);
    Result->Add(*Biases, *z);
    activation->FeedForward(z, forwardOutputDesc, Result, forwardOutputDesc);

    return Result;
}

const MAT* FCL::BackPropagate(const Matrix_GPU* lastDelta, const Matrix_GPU* PastActivation)
{
    //newDelta->Flatten();
    activation->Derivative(Result, forwardOutputDesc, lastDelta, forwardOutputDesc, z, forwardOutputDesc,
                           deltaActivation, forwardOutputDesc);
    //deltaActivation->operator*=(lastDelta); // This is done in the previous line
    DeltaBiases->Add(*deltaActivation, *DeltaBiases);

    deltaActivation->MultiplyByTransposeAndAddToRes(*PastActivation, *Delta);
    Weights->MultiplyTransposeBy(*deltaActivation, *newDelta);

    return newDelta;
}

const Matrix_GPU* FCL::getResult() const
{
    return Result;
}

const Matrix_GPU* FCL::getDelta()
{
    return Delta;
}

const Matrix_GPU* FCL::getDeltaBiases()
{
    return DeltaBiases;
}

FCL* FCL::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount), sizeof(int));
    Matrix* weights_CPU = Matrix::Read(reader);
    Matrix* biases_CPU = Matrix::Read(reader);
    Matrix_GPU* weights = new Matrix_GPU(*weights_CPU);
    Matrix_GPU* biases = new Matrix_GPU(*biases_CPU);
    Activation* activation = Activation::Read(reader);
    return new FCL(neuronsCount, activation, weights, biases, nullptr, nullptr);
}

#else

/*
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
FCL<activation,prevLayerShape,layershape,optimizer,test>::FCL(MAT* weights, MAT* bias, MAT* delta, MAT* deltaBiases)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    Weights = weights;
    Biases = bias;
}
*/

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
MAT<layershape::x>* FCL<activation,prevLayerShape,layershape,optimizer,test>::FeedForward(const MAT<prevLayerShape::x>* input)
{
    //TODO: Check a solution to flatten without copying
    //input->Flatten();
    this->Weights->MatrixMultiplication(input, Result);
    Result->Add(Biases, z);
    activation::FeedForward(z, Result);

    return Result;
}

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<prevLayerShape::x>* FCL<activation,prevLayerShape,layershape,optimizer,test>::BackPropagate(const MAT<layershape::x>* lastDelta, const MAT<prevLayerShape::x>* PastActivation)
{
    //newDelta->Flatten();
    activation::Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);

    DeltaBiases->Add(deltaActivation, DeltaBiases);

    auto* d2 = new MAT<layershape::x,prevLayerShape::x>();
    MAT<1,prevLayerShape::x>* PastActivationT = PastActivation->Transpose();
    deltaActivation->MatrixMultiplication(PastActivationT, d2);
    Delta->Add(d2, Delta);
    delete d2;
    delete PastActivationT;

    Matrix<prevLayerShape::x, layershape::x>* weightsT = Weights->Transpose();
    weightsT->MatrixMultiplication(deltaActivation, newDelta);
    delete weightsT;

    return newDelta;
}

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<layershape::x, prevLayerShape::x>* FCL<activation, prevLayerShape, layershape, optimizer,test>::BackPropagateSSE2(
    const MAT<layershape::x>* lastDelta, const MAT<layershape::x>* PastActivation)
{
    /*newDelta->Flatten();
    activation->Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);

    DeltaBiases->Add(deltaActivation, DeltaBiases);
    float* weigthsData = Weights->GetData();
    float* DeltaData = Delta->GetData();
    float* deltaActivationData = deltaActivation->GetData();
    float* newDeltaData = newDelta->GetData();

    for (int i = 0; i < previousNeuronsCount; i++)
    {
        int j = 0;
        newDeltaData[i] = 0;

        __m128 m_newDelta = _mm_setzero_ps();
        __m128 m_PastActivation = _mm_set1_ps((*PastActivation)[i]);
        int columnSize = i * NeuronsCount;
        for (j; j + 4 < NeuronsCount; j += 4)
        {
            //
            __m128 m_deltaActivation = _mm_load_ps(deltaActivationData + j);
            __m128 m_Weigths = _mm_loadu_ps(weigthsData + columnSize);


            m_newDelta = _mm_add_ps(m_newDelta, _mm_mul_ps(m_deltaActivation, m_Weigths));

            __m128 m_delta = _mm_set_ps(DeltaData[i + (j + 3) * previousNeuronsCount],
                                        DeltaData[i + (j + 2) * previousNeuronsCount],
                                        DeltaData[i + (j + 1) * previousNeuronsCount],
                                        DeltaData[i + j * previousNeuronsCount]);

            m_delta = _mm_add_ps(m_delta, _mm_mul_ps(m_PastActivation, m_deltaActivation));

            _mm_storeu_ps(buffer, m_delta);
            for (int k = 0; k < 4; k++)
            {
                DeltaData[i + (j + k) * previousNeuronsCount] = buffer[k];
            }

        }
        m_newDelta = _mm_hadd_ps(m_newDelta, m_newDelta);
        m_newDelta = _mm_hadd_ps(m_newDelta, m_newDelta);
        _mm_store_ps(buffer, m_newDelta);
        newDeltaData[i] = buffer[0] + newDeltaData[i];


        for (; j < NeuronsCount; j++)
        {
            newDeltaData[i] += deltaActivationData[j] * weigthsData[j + i * NeuronsCount];
            DeltaData[i + j * previousNeuronsCount] += PastActivation[0][i] * deltaActivationData[j];
        }

    }


    return newDelta;*/
    return nullptr;
}

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<layershape::x, prevLayerShape::x>* FCL<activation, prevLayerShape, layershape, optimizer,test>::BackPropagateAX2(
    const MAT<layershape::x>* lastDelta, const MAT<layershape::x>* PastActivation)
{

    newDelta->Flatten();
    activation::Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);

    DeltaBiases->Add(deltaActivation, DeltaBiases);
    float* weigthsData = Weights->GetData();
    float* DeltaData = Delta->GetData();

    for (int i = 0; i < previousNeuronsCount; i++)
    {
        int j = 0;
        (*newDelta)[i] = 0;

        /*
        __m256 m_newDelta = _mm256_setzero_ps();
        __m256 m_PastActivation = _mm256_set1_ps((*PastActivation)[i]);
        int columnSize = i * NeuronsCount;
        for (j; j + 8 < NeuronsCount; j+=8)
        {
            __m256 m_deltaActivation = _mm256_load_ps(&((*deltaActivation)[j]));
            __m256 m_Weigths = _mm256_loadu_ps(weigthsData + columnSize);


            m_newDelta = _mm256_add_ps(m_newDelta,_mm256_mul_ps(m_deltaActivation,m_Weigths));

            __m256 m_delta = _mm256_set_ps(DeltaData[i + (j+7) * previousNeuronsCount],DeltaData[i + (j+6)*previousNeuronsCount],DeltaData[i + (j+5)*previousNeuronsCount],DeltaData[i + (j+4)*previousNeuronsCount],DeltaData[i+ (j+3)*previousNeuronsCount],DeltaData[i+ (j+2)*previousNeuronsCount],DeltaData[i+ (j+1)*previousNeuronsCount],DeltaData[i+ j*previousNeuronsCount]);

            m_delta = _mm256_add_ps(m_delta,_mm256_mul_ps(m_PastActivation,m_deltaActivation));

            _mm256_storeu_ps(buffer,m_delta);
            for (int k = 0; k < 8; k++)
            {
                Delta[0][i + (j + k) * previousNeuronsCount] = buffer[k];
            }

        }
        m_newDelta = _mm256_hadd_ps(m_newDelta,m_newDelta);
        m_newDelta = _mm256_hadd_ps(m_newDelta,m_newDelta);
        _mm256_storeu_ps(buffer,m_newDelta);
        newDelta[0][i] = buffer[0] + buffer[1] + newDelta[0][i];

        */
        for (; j < layershape::x; j++)
        {
            newDelta[0][i] += deltaActivation[0][j] * Weights[0][j + i * layershape::x];
            Delta[0][i + j * previousNeuronsCount] += PastActivation[0][i] * deltaActivation[0][j];
        }

    }

    return newDelta;
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<layershape::x>* FCL<activation,prevLayerShape,layershape,optimizer,test>::getResult() const
{
    return Result;
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<layershape::x, prevLayerShape::x>* FCL<activation,prevLayerShape,layershape,optimizer,test>::getDelta()
{
    return Delta;
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
const MAT<layershape::x>* FCL<activation,prevLayerShape,layershape,optimizer,test>::getDeltaBiases()
{
    return DeltaBiases;
}

/*
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
FCL* FCL<activation,prevLayerShape,layershape,optimizer,test>::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount), sizeof(int));
    Matrix* weights = Matrix::Read(reader);
    Matrix* biases = Matrix::Read(reader);
    Activation* activation = Activation::Read(reader);
    return new FCL(neuronsCount, activation, weights, biases, nullptr, nullptr);
}
*/

#endif

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::ClearDelta()
{
    Delta->Zero();
    DeltaBiases->Zero();
}

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer::Compute(Delta, Weights);
    optimizer::Compute(DeltaBiases, Biases, Weights->GetSize());

    Delta->Zero();
    DeltaBiases->Zero();
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::AddDeltaFrom(Layer<FCL<activation,prevLayerShape,layershape,optimizer,test>>* otherLayer)
{
#if USE_GPU
    throw std::runtime_error("FCL::AddDeltaFrom not implemented for GPU");
#else
    FCL* _FCLLayer = (FCL*) otherLayer;
    Delta->AddAllDims(_FCLLayer->Delta, Delta);
    DeltaBiases->AddAllDims(_FCLLayer->DeltaBiases, DeltaBiases);
#endif
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
std::string FCL<activation,prevLayerShape,layershape,optimizer,test>::getLayerTitle()
{
    std::string buf;
    buf += "Layer : Fully Connected\n";
    buf += "Activation Function : " + activation::getName() + "\n";
    buf += "Neurons Count : " + std::to_string(layershape::x) + "\n";
    return buf;
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
Layer<FCL<activation,prevLayerShape,layershape,optimizer,test>>* FCL<activation,prevLayerShape,layershape,optimizer,test>::Clone()
{
    return new FCL<activation,prevLayerShape,layershape,optimizer,test>();
}

/*
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::SpecificSave(std::ofstream& write)
{
    write.write(reinterpret_cast<char*>(&NeuronsCount), sizeof(int));
    Weights->Save(write);
    Biases->Save(write);
    activation->Save(write);
}
*/

template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::AverageGradients(int batchSize)
{
    Delta->DivideAllDims(batchSize);
    DeltaBiases->DivideAllDims(batchSize);
}
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
FCL<activation,prevLayerShape,layershape,optimizer,test>::~FCL()
{
    delete Weights;
    delete Biases;
    delete Delta;
    delete DeltaBiases;
    delete deltaActivation;
    delete Result;
    delete z;
    delete newDelta;

#if USE_GPU
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardInputDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(forwardOutputDesc));
#endif
}

#if USE_GPU

void FCL::Save(const std::string& folderPath, const int n)
{
    std::ofstream weightsWriter(folderPath + "/weights" + std::to_string(n) + ".txt");
    std::ofstream biasesWriter(folderPath + "/biases" + std::to_string(n) + ".txt");
    std::ofstream deltaWriter(folderPath + "/delta" + std::to_string(n) + ".txt");
    std::ofstream deltaBiasesWriter(folderPath + "/deltaBiases" + std::to_string(n) + ".txt");
    Weights->Save(weightsWriter);
    Biases->Save(biasesWriter);
    Delta->Save(deltaWriter);
    DeltaBiases->Save(deltaBiasesWriter);
    weightsWriter.close();
    biasesWriter.close();
    deltaWriter.close();
    deltaBiasesWriter.close();
}

#else
/*
template<typename activation,typename prevLayerShape,typename layershape,typename optimizer,bool test>
void FCL<activation,prevLayerShape,layershape,optimizer,test>::Compare(const std::string& folderPath, int n)
{
    std::ifstream weightsReader(folderPath + "/weights" + std::to_string(n) + ".txt");
    std::ifstream biasesReader(folderPath + "/biases" + std::to_string(n) + ".txt");
    std::ifstream deltaReader(folderPath + "/delta" + std::to_string(n) + ".txt");
    std::ifstream deltaBiasesReader(folderPath + "/deltaBiases" + std::to_string(n) + ".txt");
    Matrix* weights = Matrix::Read(weightsReader);
    Matrix* biases = Matrix::Read(biasesReader);
    Matrix* delta = Matrix::Read(deltaReader);
    Matrix* deltaBiases = Matrix::Read(deltaBiasesReader);
    weightsReader.close();
    biasesReader.close();
    deltaReader.close();
    deltaBiasesReader.close();
    for (int i = 0; i < Weights[0].GetSize(); i++)
    {
        if (std::abs(Weights[0][i] - weights[0][i]) > 0.0001)
        {
            std::cout << "Weights[" << i << "] : " << Weights[0][i] << " != " << weights[0][i] << "\n";
        }
    }
    for (int i = 0; i < Biases[0].GetSize(); i++)
    {
        if (std::abs(Biases[0][i] - biases[0][i]) > 0.0001)
        {
            std::cout << "Biases[" << i << "] : " << Biases[0][i] << " != " << biases[0][i] << "\n";
        }
    }
    for (int i = 0; i < Delta[0].GetSize(); i++)
    {
        if (std::abs(Delta[0][i] - delta[0][i]) > 0.0001)
        {
            std::cout << "Delta[" << i << "] : " << Delta[0][i] << " != " << delta[0][i] << "\n";
        }
    }
    for (int i = 0; i < DeltaBiases[0].GetSize(); i++)
    {
        if (std::abs(DeltaBiases[0][i] - deltaBiases[0][i]) > 0.0001)
        {
            std::cout << "DeltaBiases[" << i << "] : " << DeltaBiases[0][i] << " != " << deltaBiases[0][i] << "\n";
        }
    }
    delete weights;
    delete biases;
    delete delta;
    delete deltaBiases;
}
*/

#endif


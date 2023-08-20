#include "FCL.h"
#include <iostream>
#include <cmath>
#include <emmintrin.h>
#include <immintrin.h>
#include <fstream>
#include "Matrix.h"
#include "LayerShape.h"


FCL::FCL(const int NeuronsCount, Activation* activation)
{
    this->NeuronsCount = NeuronsCount;
    this->activation = activation;
    LayerID = 0;
}

FCL::FCL(int NeuronsCount, Activation* _activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    this->NeuronsCount = NeuronsCount;
    this->activation = _activation;
    Weights = weights;
    Biases = bias;
}

void FCL::ClearDelta()
{
    Delta->Zero();
    DeltaBiases->Zero();
}

Matrix* FCL::FeedForward(const Matrix* input)
{
    input->Flatten();
    this->Weights->CrossProduct(input, Result);
    Result->Add(Biases, z);
    activation->FeedForward(z, Result);
    return Result;
}

void FCL::Compile(LayerShape* previousLayer)
{
    buffer = new float[8];
    if (previousLayer->size != 1)
    {
        throw std::invalid_argument("Previous Layer must have one dimension ! ");
    }
    previousNeuronsCount = previousLayer->dimensions[0];
    if (Weights == nullptr)
    {
        Weights = activation->InitWeights(previousNeuronsCount, NeuronsCount);
    }
    if (Delta == nullptr)
        Delta = new Matrix(previousNeuronsCount, NeuronsCount);
    if (deltaActivation == nullptr)
        deltaActivation = new Matrix(NeuronsCount, 1,1,true);
    if (DeltaBiases == nullptr)
        DeltaBiases = new Matrix(NeuronsCount, 1);
    if (Biases == nullptr)
        Biases = activation->InitBiases(NeuronsCount);
    if (Result == nullptr)
        Result = new Matrix(NeuronsCount, 1);
    z = new Matrix(NeuronsCount, 1);
    newDelta = new Matrix(previousNeuronsCount, 1);
    

    layerShape = new LayerShape(NeuronsCount);
    optimizer->Compile(NeuronsCount * previousNeuronsCount + NeuronsCount);
}

const Matrix* FCL::BackPropagate(const Matrix* lastDelta, const Matrix* PastActivation)
{
    //newDelta->Flatten();
    activation->Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);

    DeltaBiases->Add(deltaActivation, DeltaBiases);

    Matrix* d2 = new Matrix(Delta->getRows(), Delta->getCols(), Delta->getDim());
    Matrix* PastActivationT = PastActivation->Transpose();
    deltaActivation->CrossProduct(PastActivationT, d2);
    Delta->Add(d2, Delta);
    delete d2;
    delete PastActivationT;

    Matrix* weightsT = Weights->Transpose();
    weightsT->CrossProduct(deltaActivation, newDelta);
    delete weightsT;

    return newDelta;
}


const Matrix* FCL::BackPropagateSSE2(const Matrix* lastDelta,const Matrix* PastActivation)
{

    newDelta->Flatten();
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
        for (j; j + 4 < NeuronsCount; j+=4)
        {
            //
            __m128 m_deltaActivation = _mm_load_ps(deltaActivationData + j);
            __m128 m_Weigths = _mm_loadu_ps(weigthsData + columnSize);


            m_newDelta = _mm_add_ps(m_newDelta,_mm_mul_ps(m_deltaActivation,m_Weigths));
            
            __m128 m_delta = _mm_set_ps(DeltaData[i + (j+3) * previousNeuronsCount],DeltaData[i + (j+2)*previousNeuronsCount],DeltaData[i + (j+1)*previousNeuronsCount],DeltaData[i + j*previousNeuronsCount]);

            m_delta = _mm_add_ps(m_delta,_mm_mul_ps(m_PastActivation,m_deltaActivation));

            _mm_storeu_ps(buffer,m_delta);
            for (int k = 0; k < 4; k++)
            {
                DeltaData[i + (j + k) * previousNeuronsCount] = buffer[k];
            }
            
        }
        m_newDelta = _mm_hadd_ps(m_newDelta,m_newDelta);
        m_newDelta = _mm_hadd_ps(m_newDelta,m_newDelta);
        _mm_store_ps(buffer,m_newDelta);
        newDeltaData[i] = buffer[0] + newDeltaData[i];


        for (; j < NeuronsCount; j++)
        {
            newDeltaData[i] += deltaActivationData[j] * weigthsData[j + i * NeuronsCount];
            DeltaData[i + j * previousNeuronsCount] += PastActivation[0][i] * deltaActivationData[j];
        }

    }


    
    return newDelta;

}

const Matrix* FCL::BackPropagateAX2(const Matrix* lastDelta, const Matrix* PastActivation)
{
    newDelta->Flatten();
    activation->Derivative(z, deltaActivation);
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
        for (; j < NeuronsCount; j++)
        {
            newDelta[0][i] += deltaActivation[0][j] * Weights[0][j + i * NeuronsCount];
            Delta[0][i + j * previousNeuronsCount] += PastActivation[0][i] * deltaActivation[0][j];
        }

    }
    
    return newDelta;
} 






void FCL::UpdateWeights(double learningRate, int batchSize)
{
    optimizer->Compute(Delta, Weights);
    optimizer->Compute(DeltaBiases, Biases, Weights->size());

    Delta->Zero();
    DeltaBiases->Zero();
}


void FCL::AddDeltaFrom(Layer* otherLayer)
{

    FCL* _FCLLayer = (FCL*) otherLayer;
    Delta->AddAllDims(_FCLLayer->Delta, Delta);
    DeltaBiases->AddAllDims(_FCLLayer->DeltaBiases, DeltaBiases);
}

const Matrix* FCL::getResult() const
{
    return Result;
}

const Matrix* FCL::getDelta()
{
    return Delta;
}

const Matrix* FCL::getDeltaBiases()
{
    return DeltaBiases;
}

std::string FCL::getLayerTitle()
{
    std::string buf;
    buf += "Layer : Fully Connected Layer\n";
    buf += "Activation Function : " + activation->getName() + "\n";
    buf += "Neurons Count : " + std::to_string(NeuronsCount) + "\n";
    return buf;
}

Layer* FCL::Clone()
{
    return new FCL(NeuronsCount, activation, Weights->CopyWithSameData(), Biases->CopyWithSameData(), nullptr, nullptr);
}

FCL* FCL::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount), sizeof(int));
    Matrix* weights = Matrix::Read(reader);
    Matrix* biases = Matrix::Read(reader);
    Activation* activation = Activation::Read(reader);
    return new FCL(neuronsCount, activation, weights, biases, nullptr, nullptr);
}

void FCL::SpecificSave(std::ofstream& write)
{
    write.write(reinterpret_cast<char*>(&NeuronsCount), sizeof(int));
    Weights->Save(write);
    Biases->Save(write);
    activation->Save(write);
}

void FCL::AverageGradients(int batchSize)
{
    Delta->DivideAllDims(batchSize);
    DeltaBiases->DivideAllDims(batchSize);
}

FCL::~FCL()
{
    delete Weights;
    delete Biases;
    delete Delta;
    delete DeltaBiases;
    delete deltaActivation;
    delete Result;
    delete z;
    delete newDelta;
}



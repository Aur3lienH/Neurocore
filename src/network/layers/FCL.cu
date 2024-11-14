#include "network/layers/FCL.cuh"
#include <iostream>
#include <fstream>
#include "matrix/Matrix.cuh"
#include "network/LayerShape.cuh"


FCL::FCL(const int NeuronsCount, Activation* activation)
{
    this->NeuronsCount = NeuronsCount;
    this->activation = activation;
    LayerID = 0;
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
        Delta = new MAT(NeuronsCount, previousNeuronsCount);
    if (deltaActivation == nullptr)
#if USE_GPU
        deltaActivation = new Matrix_GPU(NeuronsCount, 1);
#else
        deltaActivation = new Matrix(NeuronsCount, 1, 1, true);
#endif
    if (DeltaBiases == nullptr)
        DeltaBiases = new MAT(NeuronsCount, 1);
    if (Biases == nullptr)
        Biases = activation->InitBiases(NeuronsCount);
    if (Result == nullptr)
        Result = new MAT(NeuronsCount, 1);
    z = new MAT(NeuronsCount, 1);
    newDelta = new MAT(previousNeuronsCount, 1);

    layerShape = new LayerShape(NeuronsCount);
    optimizer->Compile(NeuronsCount * previousNeuronsCount + NeuronsCount);

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

FCL::FCL(int NeuronsCount, Activation* _activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases)
{
    this->Delta = delta;
    this->DeltaBiases = deltaBiases;
    this->NeuronsCount = NeuronsCount;
    this->activation = _activation;
    Weights = weights;
    Biases = bias;
}

Matrix* FCL::FeedForward(const Matrix* input)
{
    input->Flatten();
    this->Weights->MatrixMultiplication(input, Result);
    Result->Add(Biases, z);
    activation->FeedForward(z, Result);

    return Result;
}

const Matrix* FCL::BackPropagate(const Matrix* lastDelta, const Matrix* PastActivation)
{
    newDelta->Flatten();
    activation->Derivative(z, deltaActivation);
    deltaActivation->operator*=(lastDelta);

    DeltaBiases->Add(deltaActivation, DeltaBiases);

    Matrix* d2 = new Matrix(Delta->GetRows(), Delta->GetCols(), Delta->GetDims());
    Matrix* PastActivationT = PastActivation->Transpose();
    deltaActivation->MatrixMultiplication(PastActivationT, d2);
    Delta->Add(d2, Delta);
    delete d2;
    delete PastActivationT;

    Matrix* weightsT = Weights->Transpose();
    weightsT->MatrixMultiplication(deltaActivation, newDelta);
    delete weightsT;

    return newDelta;
}

const Matrix* FCL::BackPropagateSSE2(const Matrix* lastDelta, const Matrix* PastActivation)
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

FCL* FCL::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount), sizeof(int));
    Matrix* weights = Matrix::Read(reader);
    Matrix* biases = Matrix::Read(reader);
    Activation* activation = Activation::Read(reader);
    return new FCL(neuronsCount, activation, weights, biases, nullptr, nullptr);
}

#endif


void FCL::ClearDelta()
{
    Delta->Zero();
    DeltaBiases->Zero();
}


void FCL::UpdateWeights(const double learningRate, const int batchSize)
{
    optimizer->Compute(Delta, Weights);
    optimizer->Compute(DeltaBiases, Biases, Weights->GetSize());

    Delta->Zero();
    DeltaBiases->Zero();
}

void FCL::AddDeltaFrom(Layer* otherLayer)
{
#if USE_GPU
    throw std::runtime_error("FCL::AddDeltaFrom not implemented for GPU");
#else
    FCL* _FCLLayer = (FCL*) otherLayer;
    Delta->AddAllDims(_FCLLayer->Delta, Delta);
    DeltaBiases->AddAllDims(_FCLLayer->DeltaBiases, DeltaBiases);
#endif
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

void FCL::Compare(const std::string& folderPath, int n)
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

#endif

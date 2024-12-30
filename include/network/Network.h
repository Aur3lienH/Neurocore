#pragma once

#include <iostream>
#include "network/layers/Layer.cuh"
#include "network/layers/InputLayer.cuh"
#include "matrix/Matrix.cuh"
#include "datasetsBehaviour/DataLoader.h"
#include <mutex>
#include <tools/ProgressBar.h>


template<typename Loss,typename... Layers>
class Network
{
public:
    explicit Network(Layers... args);

    ~Network();

    explicit Network(Network* network);

    //Add a layer to the network
    //void AddLayer(Layer* layer);

    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(MAT* input, MAT* output);

    void Learn(int epochs, double learningRate, MAT** inputs, MAT** outputs, int dataLength);

    //Compute a value threw the neural network
    MAT* Process(MAT* input);

    template<size_t I>
    constexpr const MAT* FeedForwardUnroll(const MAT *input);

    //Compute values through the neural network
    const MAT* FeedForward(MAT* input);

    constexpr const


    /// @brief Multithreading learning
    /// @param epochs Number of times which the neural network will see the dataset
    /// @param learningRate
    /// @param inputs The inputs of the dataset
    /// @param outputs The outputs of the dataset
    /// @param batchSize The number of turn before updating weights
    /// @param dataLength The GetSize of the dataset
    /// @param threadNumber The number of thread used to train the model
    void Learn(int epochs, double learningRate, DataLoader* dataLoader, int batchSize, int threadNumber);

    //Clear all delta from all layers (partial derivative)
    void ClearDelta();

    void PrintNetwork();

    //Initialize variable and check for error in the architecture of the model
    void Compile(Opti opti = Opti::Constant, Loss* loss = nullptr);

    //Load network from a file
    static Network* Load(const std::string& filename);

    //Save network to a file
    void Save(const std::string& filename);

private:
    void UpdateWeights(double learningRate, int batchSize);

    //Compute values and loss
    double FeedForward(MAT* input, MAT* desiredOutput);

    //The output of the network
    const MAT* output = nullptr;

    //The partial derivative of the cost
    MAT* costDerivative = nullptr;

    std::tuple<Layers...> Layers;

    //Loss function
    Loss* loss = nullptr;

    //Is Compiled ?
    bool compiled = false;

    //The number of layers
    int layersCount = 0;

    Opti opti;

public:
    static inline int ctr = 0;
};




template<typename Loss,typename... Types>
Network<Loss,Types...>::Network(Types... Args)
{

	this->Layers = nullptr;
}

template<typename Loss,typename... Types>
Network<Loss,Types...>::Network(Network* network)
{
    this->layersCount = network->layersCount;
    this->Layers = new Layer* [layersCount];
    for (int i = 0; i < layersCount; i++)
        this->Layers[i] = network->Layers[i]->Clone();

}


template<typename Loss,typename... Layers>
void
#if USE_GPU
LearnThread2(Network* network, Matrix_GPU*** data, const int batchSize, const int batchIndex,
             const int numDataPerThread)
#else
LearnThread2(Network<Loss,Layers>* network, Matrix*** data, const int batchSize, const int batchIndex, const int numDataPerThread)
#endif
{
    //std::cout<<j<<" " << threadArg->batchSize<<std::endl;
    //std::cout<< "Aux = [" << j * threadArg->batchSize << ";" << j * threadArg->batchSize + numDataPerThread - 1 << "]" << std::endl;
    network->ClearDelta();
    for (int i = 0; i < numDataPerThread; i++)
    {
        const int dataIndex = batchIndex * batchSize + i;
        network->BackPropagate(data[dataIndex][0], data[dataIndex][1]);
    }
    //std::cout<<"batch " << j << " finished\n";
}
template<typename Loss,typename... Layers>
MAT* Network<Loss,Layers...>::Process(MAT* input)
{
    const MAT* res = FeedForward(input);
    return const_cast<MAT*>(res);
}

template<typename Loss,typename... Layers>
template<size_t I>
constexpr const MAT* Network<Loss,Layers...>::FeedForwardUnroll(const MAT* input)
{
    if constexpr (I == sizeof...(Layers))
    {
        return input;
    }
    else
    {
        auto& layer = std::get<I>(Layers);
        return FeedForwardUnroll<I + 1>(layer.FeedForward(input));
    }
}

template<typename Loss,typename... Layers>
constexpr const MAT* Network<Loss,Layers...>::FeedForward(MAT* input)
{
    output = FeedForwardUnroll<0>(input);
    return output;
}

template<typename Loss,typename... Layers>
double Network<Loss,Layers...>::FeedForward(MAT* input, MAT* desiredOutput)
{

    output = input;
    for (int i = 0; i < layersCount; i++)
    {
        //std::cout << "feedforward : " << i << "\n";
        output = Layers[i]->FeedForward(output);
    }
    //std::cout << *output;
    return loss->Cost(output, desiredOutput);
}


//Prepare all the elements of the network before using it.

template<typename Loss,typename... Layers>
void Network<Loss,Layers...>::Compile(Opti _opti, Loss* _loss)
{
    opti = _opti;
    if (_loss != nullptr)
        loss = _loss;


    //Cannot compile a network which has no loss function
    if (loss == nullptr)
        throw std::invalid_argument("Must have a lost function !");

    //All network need to have more than Players
    if (layersCount < 2)
        throw std::invalid_argument("Network must have at least 2 layers");
    std::cout << "verifications done \n";

    std::cout << layersCount << " this is the layers Count\n";

    //Compile each layer
    for (int i = 0; i < layersCount; i++)
    {
	std::cout << i << "\n";
        if (i == 0)
            Layers[i]->Compile(nullptr, opti);
        else
            Layers[i]->Compile(Layers[i - 1]->GetLayerShape(), opti);
    }

    std::cout << "loop done \n";

    //Initialize the matrix which holds the values of the cost derivative.
    costDerivative = Layers[layersCount - 1]->GetLayerShape()->ToMatrix();

    std::cout << "Getting cost derivative matrix done \n";
    auto* inputLayer = (Layer<InputLayer>*) Layers[0];
    if (inputLayer == nullptr)
        throw std::invalid_argument("First layer must be an input layer");

    std::cout << "finished compiling \n";
    compiled = true;
}

#include "network/layers/FCL.cuh"

template<typename Loss,typename... Layers>
double Network<Loss,Layers...>::BackPropagate(MAT* input, MAT* desiredOutput)
{
/*    FCL* fcl = (FCL*) Layers[1];
#if USE_GPU
    fcl->Save("FCL", ctr);
#else
    fcl->Compare("FCL", ctr);
#endif*/

    //std::cout << "feeding forward ! \n";
    double NetworkLoss = FeedForward(input, desiredOutput);
    //if (++ctr == 2) abort();
    //std::cout << "feeding stoped ! \n";
    loss->CostDerivative(Layers[layersCount - 1]->getResult(), desiredOutput, costDerivative);
    output = costDerivative;
    for (int i = layersCount - 1; i > 0; i--)
    {
        output = Layers[i]->BackPropagate(output, Layers[i - 1]->getResult());
    }

    return NetworkLoss;
}

template<typename Loss,typename... Layers>
void Network<Loss,Layers...>::ClearDelta()
{
    for (int i = 0; i < layersCount; i++)
        Layers[i]->ClearDelta();

}

template<typename Loss,typename... Layers>
void
Network<Loss,Layers...>::Learn(const int epochs, const double learningRate, MAT** inputs, MAT** outputs,
               const int dataLength)
{
    Tools::TrainBar Bar = Tools::TrainBar(epochs * dataLength);
    double globalLoss;
    for (int e = 0; e < epochs; e++)
    {
        globalLoss = 0;
        for (int i = 0; i < dataLength; i++)
        {
            globalLoss += BackPropagate(inputs[i], outputs[i]);
            UpdateWeights(learningRate, 1);
            Bar.ChangeProgress(e * dataLength + i, globalLoss / (i + 1));
        }
    }
}

template<typename Loss,typename... Layers>
const void Network<Loss, Layers...>::Learn(const int epochs, const double learningRate, DataLoader *dataLoader,
                                           const int batchSize,
                                           const int threadNumber)
{
    //Check if the network is compiled
    if (!compiled)
        throw std::invalid_argument("Network must be compiled before learning");

    //Check if there is enough thread to have a minimum of 1 inputs per thread
    if (batchSize < threadNumber)
        throw std::invalid_argument("More thread than batch GetSize !");


    //Initializing the progress bar
    Tools::TrainBar Bar = Tools::TrainBar(epochs);

    //Number of threads excluding the main one
    const int auxThreadNumber = threadNumber - 1;
    //Number of data per thread
    const int numberPerThread = batchSize / threadNumber;
    //Number of batch per epoch
    const int numberOfBatches = dataLoader->GetSize() / batchSize;

    std::thread* threads;
    Network** auxiliaryNetworks;

    //If there are some auxiliary threads initialize variables
    if (auxThreadNumber > 0)
    {
        threads = new std::thread[auxThreadNumber];
        auxiliaryNetworks = new Network* [auxThreadNumber];
    }

    /*int rest;
    if(auxThreadNumber != 0)
    {
        rest = batchSize % auxThreadNumber;
    }
    else
    {
        rest = batchSize;
    }*/
    double globalLoss;

    //Creating all the auxiliary threads and networks
    for (int j = 0; j < auxThreadNumber; j++)
    {
        auto* NetworkCopy = new Network(this);
        NetworkCopy->Compile(opti, loss);
        auxiliaryNetworks[j] = NetworkCopy;
    }

    std::cout << "learning starting ! \n";

    // Computing data offset for each thread
    const int mainThreadDataOffset = auxThreadNumber * numberPerThread;
    int* auxThreadsDataOffset = new int[auxThreadNumber];
    for (int i = 0; i < auxThreadNumber; i++)
        auxThreadsDataOffset[i] = i * numberPerThread;

    for (int e = 0; e < epochs; e++)
    {
        //Resetting the loss for each epoch
        globalLoss = 0;

        for (int k = 0; k < numberOfBatches; k++)
        {
            // Start auxiliary threads
            for (int i = 0; i < auxThreadNumber; ++i)
            {
                threads[i] = std::thread(&LearnThread2, auxiliaryNetworks[i],
                                         dataLoader->data + auxThreadsDataOffset[i],
                                         batchSize, k, numberPerThread);
            }

            //Clear Deltatd::cout << "epochs : " << e << " numberOfbatches : " < < k << " index : " << index << "  \n";
            ClearDelta();
            //std::cout<< "Main = [" << mainThreadDataOffset + k * batchSize << ";" << mainThreadDataOffset + k * batchSize + numberPerThread - 1 << "]" << std::endl;
            for (int i = 0; i < numberPerThread; i++)
            {
                const int index = mainThreadDataOffset + k * batchSize;

                //Backpropagate in the main thread
                globalLoss += BackPropagate(dataLoader->data[index + i][0], dataLoader->data[index + i][1]);
            }
            //std::cout<<"Main batch " << k << " finished\n";

            // Sync auxiliary threads
            for (int i = 0; i < auxThreadNumber; i++)
                threads[i].join();

            //Get the delta from all threads and update weights
            for (int i = 1; i < layersCount; i++)
            {
                for (int j = 0; j < auxThreadNumber; j++)
                    Layers[i]->AddDeltaFrom(auxiliaryNetworks[j]->Layers[i]);

                Layers[i]->AverageGradients(batchSize);
                Layers[i]->UpdateWeights(learningRate, batchSize);
            }

            //Update the progress bar
            std::cout << "\repochs : " << e << " loss: " << globalLoss / ((k + 1) * batchSize) << std::flush;
            //Bar.ChangeProgress(e, globalLoss / ((k + 1) * numberPerThread));
        }

        dataLoader->Shuffle();
    }
    std::cout << "Learning finished" << std::endl;

    if (auxThreadNumber)
    {
        delete[] threads;
        delete[] auxiliaryNetworks;
        delete[] auxThreadsDataOffset;
    }
    delete dataLoader;
}
template<typename Loss,typename... Layers>
void Network<Loss,Layers...>::UpdateWeights(const double learningRate, const int batchSize)
{
    for (int i = 1; i < layersCount; i++)
        Layers[i]->UpdateWeights(learningRate, batchSize);
}

template<typename Loss,typename... Layers>
void Network<Loss,Layers...>::PrintNetwork()
{
    std::cout << " ---- Network ---- \n";
    for (int i = 0; i < layersCount; i++)
        std::cout << Layers[i]->getLayerTitle() << std::endl;
}

template<typename Loss,typename... Layers>
void Network<Loss,Layers...>::Save(const std::string& filename)
{
    std::ofstream writer;
    writer.open(filename, std::ios::binary | std::ios::trunc);

    loss->Save(writer);
    //First save the number of layers
    writer.write(reinterpret_cast<char*>(&layersCount), sizeof(int));

    for (int i = 0; i < layersCount; i++)
        Layers[i]->Save(writer);

    writer.close();
}
template<typename Loss,typename... Layers>
Network<Loss,Layers...>* Network<Loss,Layers...>::Load(const std::string& filename)
{
    std::ifstream reader;
    reader.open(filename, std::ios::binary);
    auto* network = new Network();
    //Load number of layers

    Loss* loss = Loss::Read(reader);
    int layersCount;
    reader.read(reinterpret_cast<char*>(&layersCount), sizeof(int));

    //Load each Layer
    /* TODO : Need a proper Layer loader to handle the templates
    for (int i = 0; i < layersCount; i++)
        network->AddLayer(Layer::Load(reader));
    */
    network->Compile(Opti::Constant, loss);
    reader.close();

    return network;
}


template<typename Loss,typename... Layers>
Network<Loss,Layers...>::~Network()
{
    for (int i = 0; i < layersCount; i++)
        delete Layers[i];
    /*
    if(Layers != nullptr)
    	delete[] Layers;
    */
    if(loss != NULL)
    	delete loss;
    //delete output; // It is already deleted when deleting the last layer
    if(costDerivative != nullptr)
    	delete costDerivative;
}

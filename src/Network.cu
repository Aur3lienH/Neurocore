#include "Network.cuh"
#include "InputLayer.cuh"
#include "Loss.cuh"
#include "Tools/ProgressBar.cuh"
#include "DataLoader.cuh"
#include <fstream>
#include <iostream>
#include <cmath>
#include <mutex>
#include <thread>


Network::Network()
{

}

Network::Network(Network* network)
{
    this->layersCount = network->layersCount;
    this->Layers = new Layer* [layersCount];
    for (int i = 0; i < layersCount; i++)
        this->Layers[i] = network->Layers[i]->Clone();

}

void
#if USE_GPU
LearnThread2(Network* network, Matrix_GPU*** data, const int batchSize, const int batchIndex,
             const int numDataPerThread)
#else
LearnThread2(Network* network, Matrix*** data, const int batchSize, const int batchIndex, const int numDataPerThread)
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

void Network::AddLayer(Layer* layer)
{
    if (layersCount == 0)
    {
        Layers = new Layer* [1];
        Layers[0] = layer;
        layersCount++;
    }
    else
    {
        auto** temp = new Layer* [layersCount + 1];
        for (int i = 0; i < layersCount; i++)
        {
            temp[i] = Layers[i];
        }
        temp[layersCount] = layer;
        delete[] Layers;
        Layers = temp;
        layersCount++;
    }
}

MAT* Network::Process(MAT* input)
{
    const MAT* res = FeedForward(input);
    return const_cast<MAT*>(res);
}

const MAT* Network::FeedForward(MAT* input)
{
    output = input;
    for (int i = 0; i < layersCount; i++)
    {
        output = Layers[i]->FeedForward(output);
    }

    return output;
}

double Network::FeedForward(MAT* input, MAT* desiredOutput)
{
    output = input;
    for (int i = 0; i < layersCount; i++)
    {
        //std::cout << "feedforward : " << i << "\n";
        output = Layers[i]->FeedForward(output);
    }

    return loss->Cost(output, desiredOutput);
}


//Prepare all the elements of the network before using it.

void Network::Compile(Opti _opti, Loss* _loss)
{
    opti = _opti;

    if (_loss != nullptr)
        loss = _loss;


    //Cannot compile a network which has no loss function
    if (loss == nullptr)
        throw std::invalid_argument("Must have a lost function !");

    //All network need to have more than 2 layers
    if (layersCount < 2)
        throw std::invalid_argument("Network must have at least 2 layers");


    //Compile each layer
    for (int i = 0; i < layersCount; i++)
    {
        if (i == 0)
            Layers[i]->Compile(nullptr, opti);
        else
            Layers[i]->Compile(Layers[i - 1]->GetLayerShape(), opti);
    }

    //Initialize the matrix which holds the values of the cost derivative.
    costDerivative = Layers[layersCount - 1]->GetLayerShape()->ToMatrix();


    auto* inputLayer = (InputLayer*) Layers[0];
    if (inputLayer == nullptr)
        throw std::invalid_argument("First layer must be an input layer");

    compiled = true;

}

#include "FCL.cuh"

double Network::BackPropagate(MAT* input, MAT* desiredOutput)
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
        //std::cout << "i : " << i << "\n";
        output = Layers[i]->BackPropagate(output, Layers[i - 1]->getResult());
    }

    return NetworkLoss;
}

void Network::ClearDelta()
{
    for (int i = 0; i < layersCount; i++)
        Layers[i]->ClearDelta();

}

void

Network::Learn(const int epochs, const double learningRate, MAT** inputs, MAT** outputs,
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

void Network::Learn(const int epochs, const double learningRate, DataLoader* dataLoader, const int batchSize,
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
    const int numberOfBatches = dataLoader->dataLength / batchSize;

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

void Network::UpdateWeights(const double learningRate, const int batchSize)
{
    for (int i = 1; i < layersCount; i++)
        Layers[i]->UpdateWeights(learningRate, batchSize);
}


void Network::PrintNetwork()
{
    for (int i = 0; i < layersCount; i++)
        std::cout << Layers[i]->getLayerTitle() << std::endl;
}


void Network::Save(const std::string& filename)
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

Network* Network::Load(const std::string& filename)
{
    std::ifstream reader;
    reader.open(filename, std::ios::binary);
    auto* network = new Network();
    //Load number of layers

    Loss* loss = Loss::Read(reader);
    int layersCount;
    reader.read(reinterpret_cast<char*>(&layersCount), sizeof(int));

    //Load each Layer
    for (int i = 0; i < layersCount; i++)
        network->AddLayer(Layer::Load(reader));

    network->Compile(Opti::Constant, loss);
    reader.close();

    return network;
}

Network::~Network()
{
    for (int i = 0; i < layersCount; i++)
        delete Layers[i];

    delete[] Layers;
    delete loss;
    //delete output; // It is already deleted when deleting the last layer
    delete costDerivative;
}

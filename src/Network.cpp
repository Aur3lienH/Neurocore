#include "Network.h"
#include "InputLayer.h"
#include "LastLayer.h"
#include "Loss.h"
#include "ThreadArg.h"
#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <math.h>
#include <mutex>
#include <condition_variable>


Network::Network()
{
    
}

Network::Network(Network* network, Matrix** deltas, Matrix** deltaBiases)
{
    this->layersCount = network->layersCount;
    this->Layers = new Layer*[layersCount];
    for(int i = 0; i < layersCount; i++)
    {
        this->Layers[i] = network->Layers[i]->Clone(deltas[i], deltaBiases[i]);
    }
}

void* Network::LearnThread(void* args)
{
    ThreadArg* threadArg = (ThreadArg*)args;
    while (true)
    {
        std::unique_lock<std::mutex> lock(*threadArg->mutex);
        threadArg->cv->wait(lock);
        std::cout << "Thread started\n";
        threadArg->network->ClearDelta();
        for(int i = 0; i < threadArg->dataLength;i++)
        {
            threadArg->network->BackPropagate((*threadArg->inputs)[i], (*threadArg->outputs)[i]);
        }
        // Wait for the main thread to signal
    }
}

void Network::AddLayer(Layer* layer)
{


    if (layersCount == 0)
    {
        Layers = new Layer*[1];
        Layers[0] = layer;
        layersCount++;
    }
    else
    {
        Layer** temp = new Layer*[layersCount + 1];
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



Matrix* Network::FeedForward(Matrix* input) 
{
    output = input;
    for (int i = 0; i < layersCount; i++)
    {
        output = Layers[i]->FeedForward(output);
    }
    return output;
}

double Network::FeedForward(Matrix* input, Matrix* desiredOutput)
{
    output = input;
    for (int i = 0; i < layersCount-1; i++)
    {
        output = Layers[i]->FeedForward(output);
    }
    return lastLayer->FeedForward(output, desiredOutput);
}

void Network::Compile()
{
    

    if(layersCount < 2)
    {
        throw std::invalid_argument("Network must have at least 2 layers");
    }
    for (int i = 0; i < layersCount; i++)
    {
        if(i == 0)
            Layers[i]->Compile(0);
        else
            Layers[i]->Compile(Layers[i - 1]->getNeuronsCount(0));
    }
    InputLayer* inputLayer = (InputLayer*)Layers[0];
    if(inputLayer == nullptr)
    {
        throw std::invalid_argument("First layer must be an input layer");
    }

    lastLayer = (LastLayer*)Layers[layersCount - 1];
    if(lastLayer == nullptr)
    {
        throw std::invalid_argument("Last layer must be a LastLayer");
    }
    compiled = true;
    std::cout << "Network compiled" << std::endl;
}

double Network::BackPropagate(Matrix* input,Matrix* desiredOutput)
{
    double loss = FeedForward(input, desiredOutput);
    output = desiredOutput;
    for (int i = layersCount - 1; i > 0; i--)
    {
        output = Layers[i]->BackPropagate(output, Layers[i - 1]->getResult());
    }
    return loss;
}

void Network::ClearDelta()
{
    for (int i = 0; i < layersCount; i++)
    {
        Layers[i]->ClearDelta();
    }
}

void Network::Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength)
{
    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < dataLength; i++)
        {
            double loss = BackPropagate(inputs[i], outputs[i]);
            UpdateWeights(learningRate, 1);
            std::cout << "Epoch: " << e << " Loss: " << loss/(i+1);
            std::cout << "\n";
        }

    }
    
    
}

void Network::Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int batchSize,int dataLength, int threadNumber)
{
    if(!compiled)
    {
        throw std::invalid_argument("Network must be compiled before learning");
    }

    int pos = 0;
    int auxThreadNumber = threadNumber - 1;
    pthread_t* threads;
    Matrix*** delta;
    Matrix*** deltaBiases;
    Matrix* mainDelta;
    Matrix* mainDeltaBiases;
    std::condition_variable* cv = new std::condition_variable;
    std::mutex** mutexes;
    if(auxThreadNumber > 0)
    {
        delta = new Matrix**[auxThreadNumber];
        deltaBiases = new Matrix**[auxThreadNumber];
        mutexes = new std::mutex*[auxThreadNumber];
        threads = new pthread_t[auxThreadNumber];

    }
    for (int i = 0; i < auxThreadNumber; i++)
    {
        mutexes[i] = new std::mutex;
    }
    int numberPerThread = batchSize / threadNumber;
    int numberOfBatches = dataLength / batchSize;
    int rest;
    if(auxThreadNumber != 0)
    {
        rest = batchSize % auxThreadNumber;
    }
    else
    {
        rest = batchSize;
    }
    double globalLoss = 0;

    Matrix**** inputsPerThread = new Matrix***[threadNumber];
    Matrix**** outputsPerThread = new Matrix***[threadNumber];

    for (int i = 0; i < threadNumber; i++)
    {
        inputsPerThread[i] = new Matrix**[1];
        outputsPerThread[i] = new Matrix**[1];
    }
    
    for(int j = 0; j < auxThreadNumber; j++)
    {
        delta[j] = new Matrix*[layersCount];
        deltaBiases[j] = new Matrix*[layersCount];
        for (int i = 0; i < layersCount; i++)
        {

            if(i==0)
            {
                delta[j][i] = new Matrix(0, Layers[i]->getNeuronsCount(0));
            }
            else
            {
                delta[j][i] = new Matrix(Layers[i-1]->getNeuronsCount(0), Layers[i]->getNeuronsCount(0));
            }
            deltaBiases[j][i] = new Matrix(Layers[i]->getNeuronsCount(0), 1);
        }

        Network* NetworkCopy = new Network(this, delta[j], deltaBiases[j]);
        NetworkCopy->Compile();

        ThreadArg* threadArg = new ThreadArg(NetworkCopy, inputsPerThread[j], outputsPerThread[j], mutexes[j], cv, numberPerThread);

        pthread_create(&threads[j], NULL, LearnThread, (void*)threadArg);
    }

    for (int e = 0; e < epochs; e++)
    {
        globalLoss = 0;
        for (int k = 0; k < numberOfBatches; k++)
        {
            for (int j = 0; j < threadNumber; j++)
            {
                inputsPerThread[j][0] = inputs + pos;
                outputsPerThread[j][0] = outputs + pos;
                pos += numberPerThread;
                if(pos >= dataLength)
                {
                    pos = 0;
                }
            }
            std::cout << "pbl 1\n";
            cv->notify_all();
            std::cout << "pbl 2\n";
            ClearDelta();
            std::cout << "pbl 2.5\n";
            for (int i = 0; i < numberPerThread; i++)
            {
                globalLoss += BackPropagate(inputsPerThread[threadNumber-1][0][i], outputsPerThread[threadNumber-1][0][i]);
            }
            std::cout << "pbl 3\n";
            Matrix* a = FeedForward(inputsPerThread[threadNumber-1][0][0]);
            
            
            for (int i = 0; i < auxThreadNumber; i++)
            {
                mutexes[i]->lock();
            }
            for (int i = 1; i < layersCount; i++)
            {
                mainDelta = Layers[i]->getDelta();
                mainDeltaBiases = Layers[i]->getDeltaBiases();
                for (int j = 0; j < auxThreadNumber; j++)
                {
                    mainDelta->Add(delta[j][i], mainDelta);
                    mainDeltaBiases->Add(deltaBiases[j][i], mainDeltaBiases);
                }
                
            }
            
            
            for (int i = 1; i < layersCount; i++)
            {
                Layers[i]->UpdateWeights(learningRate, batchSize, Layers[i]->getDelta(), Layers[i]->getDeltaBiases());
            }
            for (int i = 0; i < auxThreadNumber; i++)
            {
                mutexes[i]->unlock();
            }
            std::cout << "\r";
            std::cout << "Epoch " << e << " Batch " << k << " Loss " << globalLoss / (k+1) << "                        ";
            std::cout.flush();
        }
        std::cout << std::endl;
    }
    std::cout << "Learning finished" << std::endl;
    
    /*
    for (int i = 0; i < threadNumber; i++)
    {
        pthread_join(threads[i], NULL);
    }
    */
    
}

void Network::UpdateWeights(double learningRate, int batchSize)
{
    for (int i = 1; i < layersCount; i++)
    {
        Layers[i]->UpdateWeights(learningRate, batchSize);
    }
}



void Network::PrintNetwork()
{
    for (int i = 0; i < layersCount; i++)
    {
        std::cout << Layers[i]->getLayerTitle() << std::endl;
    }
}




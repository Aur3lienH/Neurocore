#include "Network.h"
#include "InputLayer.h"
#include "Loss.h"
#include "ThreadArg.h"
#include "Tools/ProgressBar.h"
#include <fstream>
#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <math.h>
#include <mutex>
#include <condition_variable>


Network::Network()
{
    
}

Network::Network(Network* network)
{
    this->layersCount = network->layersCount;
    this->Layers = new Layer*[layersCount];
    for(int i = 0; i < layersCount; i++)
    {
        this->Layers[i] = network->Layers[i]->Clone();
    }
}

void* Network::LearnThread(void* args)
{
    ThreadArg* threadArg = (ThreadArg*)args;
    while (true)
    {
        std::unique_lock<std::mutex> lock(*threadArg->mutex);
        threadArg->cv->wait(lock);
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


Matrix* Network::Process(Matrix* input)
{
    const Matrix* res = FeedForward(input);
    return Matrix::Copy(res);
}


const Matrix* Network::FeedForward(Matrix* input) 
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
    for (int i = 0; i < layersCount; i++)
    {
        output = Layers[i]->FeedForward(output);
    }
    return loss->Cost(output,desiredOutput);
}


//Prepare all the elements of the network before using it.

void Network::Compile()
{
    //Cannot compile a network which has no loss function
    if(loss == nullptr)
    {
        throw std::invalid_argument("Must have a lost function !");
    }
    //All network need to have more than 2 layers
    if(layersCount < 2)
    {
        throw std::invalid_argument("Network must have at least 2 layers");
    }

    //Compile each layer
    for (int i = 0; i < layersCount; i++)
    {
        if(i == 0)
            Layers[i]->Compile(nullptr);
        else
            Layers[i]->Compile(Layers[i - 1]->GetLayerShape());
    }

    //Initialize the matrix which holds the values of the cost derivative.
    costDerivative = Layers[layersCount - 1]->GetLayerShape()->ToMatrix();
    InputLayer* inputLayer = (InputLayer*)Layers[0];
    if(inputLayer == nullptr)
    {
        throw std::invalid_argument("First layer must be an input layer");
    }


    compiled = true;

    std::cout << "Network compiled" << std::endl;
}

void Network::Compile(Loss* _loss)
{
    loss = _loss;
    Compile();
}

double Network::BackPropagate(Matrix* input,Matrix* desiredOutput)
{
    double loss = FeedForward(input, desiredOutput);
    this->loss->CostDerivative(Layers[layersCount - 1]->getResult(),desiredOutput,costDerivative);
    output = costDerivative;
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
    //Check if the network is compiled
    if(!compiled)
    {
        throw std::invalid_argument("Network must be compiled before learning");
    }

    //Check if there is enough thread to have a minimum of 1 inputs per thread
    if(batchSize < threadNumber)
    {
        throw std::invalid_argument("More thread than batch size !");
    }

    //Initializing the progress bar
    Tools::TrainBar Bar = Tools::TrainBar(epochs);

    //Position in the dataset
    int pos = 0;

    //Number of threads excluding the main one
    int auxThreadNumber = threadNumber - 1;

    //Array of the threads
    pthread_t* threads;

    //Waiter for auxiliary threads
    std::condition_variable* cv = new std::condition_variable;

    Network* auxiliaryNetwork;

    //Locker for threads
    std::mutex** mutexes;

    //If there is some auxiliary threads intiliaze varialbles
    if(auxThreadNumber > 0)
    {
        mutexes = new std::mutex*[auxThreadNumber];
        for (int i = 0; i < auxThreadNumber; i++)
        {
            mutexes[i] = new std::mutex;
        }
        threads = new pthread_t[auxThreadNumber];
        auxiliaryNetwork = new Network[auxThreadNumber];

    }
    //Number of data per thread
    int numberPerThread = batchSize / threadNumber;

    //Number of batch per epochs
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

    //Pointer to the dataset
    Matrix**** inputsPerThread = new Matrix***[threadNumber];
    Matrix**** outputsPerThread = new Matrix***[threadNumber];

    for (int i = 0; i < threadNumber; i++)
    {
        inputsPerThread[i] = new Matrix**[1];
        outputsPerThread[i] = new Matrix**[1];
    }
    

    //Creating all the auxiliary threads and networks
    for(int j = 0; j < auxThreadNumber; j++)
    {

        Network* NetworkCopy = new Network(this);
        NetworkCopy->Compile();

        ThreadArg* threadArg = new ThreadArg(NetworkCopy, inputsPerThread[j], outputsPerThread[j], mutexes[j], cv, numberPerThread);
        auxiliaryNetwork[j] = NetworkCopy;
        pthread_create(&threads[j], NULL, LearnThread, (void*)threadArg);
    }


    for (int e = 0; e < epochs; e++)
    {
        //Resseting the loss for each epochs
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

            //Notify all threads to start the training
            cv->notify_all();

            //Clear Delta
            ClearDelta();
            for (int i = 0; i < numberPerThread; i++)
            {
                //Backpropagate in the main thread
                globalLoss += BackPropagate(inputs[pos], outputs[pos]);
            }
            
            
            //Lock every auxiliary threads
            for (int i = 0; i < auxThreadNumber; i++)
            {
                mutexes[i]->lock();
            }
            
            
            //Get the delta from all threads and update weights
            for (int i = 1; i < layersCount; i++)
            {
                for (int j = 0; j < auxThreadNumber; j++)
                {
                    Layers[i]->AddDeltaFrom(auxiliaryNetwork[j].Layers[i]);
                }
                Layers[i]->UpdateWeights(learningRate, batchSize);
            }

            //Unlock auxiliary threads
            for (int i = 0; i < auxThreadNumber; i++)
            {
                mutexes[i]->unlock();
            }

            //Update the progress bar
            Bar.ChangeProgress(e+1, globalLoss / (k+1));
        }
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



void Network::Save(std::string filename)
{
    std::ofstream writer;
    writer.open(filename, std::ios::binary | std::ios::trunc);

    //First save the number of layers
    writer.write(reinterpret_cast<char*>(&layersCount),sizeof(int));

    for (int i = 0; i < layersCount; i++)
    {
        Layers[i]->Save(writer);
    }
    writer.close();
}

Network* Network::Load(std::string filename)
{
    std::ifstream reader;
    reader.open(filename, std::ios::binary);
    Network* network = new Network();
    //Load number of layers
    int layersCount;
    reader.read(reinterpret_cast<char*>(&layersCount),sizeof(int));

    //Load each Layer
    for (int i = 0; i < layersCount; i++)
    {
        network->AddLayer(Layer::Load(reader));
    }
    reader.close();
    return network;
}
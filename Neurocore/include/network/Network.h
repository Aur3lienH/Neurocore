#pragma once

#include <iostream>
#include "network/layers/Layer.cuh"
#include "network/layers/InputLayer.cuh"
#include "matrix/Matrix.cuh"
#include "datasetsBehaviour/DataLoader.h"
#include <mutex>
#include <tools/ProgressBar.h>




template<typename Loss,typename... Layers>
class Network final
{
public:
    Network() : layers(Layers()...) {

    }



    ~Network()
    {
        /*
        if(Layers != nullptr)
            delete[] Layers;
        */
        //delete output; // It is already deleted when deleting the last layer
        if(costDerivative != nullptr)
            delete costDerivative;
    }

    explicit Network(Network* network);

    using FirstLayer = typename std::tuple_element<0, std::tuple<Layers...>>::type;

    using LastLayer = typename std::tuple_element<sizeof...(Layers) - 1, std::tuple<Layers...>>::type;
    static constexpr std::size_t layersCount = sizeof...(Layers);

    template<std::size_t I>
    using LayersType_At = std::tuple_element_t<I, std::tuple<Layers...>>;

    using InputShape = typename FirstLayer::Shape;
    using OutputShape = typename LastLayer::Shape;

    //Add a layer to the network
    //void AddLayer(Layer* layer);

    template <typename Output,int I>
    void for_each_layer_back_prop(Output output) {
        if constexpr (I >= 1) {
            auto new_out = std::get<I>(layers).BackPropagate(output, std::get<I-1>(layers).getResult());
            for_each_layer_back_prop<decltype(new_out),I-1>(new_out);
        }
    }

    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(LMAT<InputShape>* input, LMAT<OutputShape>* desiredOutput)
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
        Loss::CostDerivative(std::get<layersCount - 1>(layers).getResult(), desiredOutput, costDerivative);
        output = costDerivative;
        for_each_layer_back_prop<decltype(output),layersCount - 1>(output);
        // for (int i = layersCount - 1; i > 0; i--)
        // {
        //     output = std::get<i>(layersCount).BackPropagate(output, std::get<layersCount - 1>(layers).getResult());
        // }

        return NetworkLoss;
    }



    void Learn(int epochs, double learningRate, DataLoader<Network>* data_loader)
    {
        Tools::TrainBar Bar = Tools::TrainBar(epochs);
        double globalLoss;
        for (int e = 0; e < epochs; e++)
        {
            globalLoss = 0;
            for (int i = 0; i < data_loader->GetSize(); i++)
            {
                globalLoss += BackPropagate(data_loader->inputs + i, data_loader->outputs + i);
                UpdateWeights(learningRate, 1);
                Bar.ChangeProgress(e+1, globalLoss / (i + 1));
            }
        }
    }

    //Compute a value threw the neural network
    LMAT<InputShape>* Process(LMAT<OutputShape>* input)
    {
        const auto* res = FeedForward(input);
        return const_cast<LMAT<OutputShape>*>(res);
    }

    template<typename InputShapeL = InputShape,int I = 0>
    auto FeedForwardRecu(const LMAT<InputShapeL>* input)
    {
        if constexpr (I == layersCount)
        {
            return input;
        }
        else
        {
            return FeedForwardRecu<typename LayersType_At<I>::Shape,I + 1>(std::get<I>(layers).FeedForward(input));
        }
    }


    //Compute values through the neural network
    const LMAT<OutputShape>* FeedForward(LMAT<InputShape>* input)
    {
        return FeedForwardRecu(input);
    }





    /// @brief Multithreading learning
    /// @param epochs Number of times which the neural network will see the dataset
    /// @param learningRate
    /// @param inputs The inputs of the dataset
    /// @param outputs The outputs of the dataset
    /// @param batchSize The number of turn before updating weights
    /// @param dataLength The GetSize of the dataset
    /// @param threadNumber The number of thread used to train the model
    void Learn(int epochs, double learningRate, DataLoader<Network>* dataLoader, int batchSize, int threadNumber)
    {
      /*
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

//        int rest;
//        if(auxThreadNumber != 0)
//        {
//            rest = batchSize % auxThreadNumber;
//        }
//        else
//        {
//            rest = batchSize;
//        }
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
                        layers[i]->AddDeltaFrom(auxiliaryNetworks[j]->layers[i]);

                    layers[i]->AverageGradients(batchSize);
                    layers[i]->UpdateWeights(learningRate, batchSize);
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
        */
    }

    //Clear all delta from all layers (partial derivative)
    void ClearDelta()
    {
        for (int i = 0; i < layersCount; i++)
            layers[i]->ClearDelta();
    }

    template<size_t I = 0>
    void PrintNetworkRecu()
    {
        if constexpr(I < layersCount)
        {
            std::cout << std::get<I>(layers).getLayerTitle() << "\n";
            PrintNetworkRecu<I+1>();
        }
    }

    void PrintNetwork()
    {
        std::cout << " ---- Network ---- \n";
        PrintNetworkRecu();
        std::cout << " ---- Network ---- \n";
    }

    template <typename Tuple,std::size_t I = 0>
    void for_each_tuple(Tuple& t) {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            std::get<I>(t).Compile();
            for_each_tuple<Tuple,I + 1>(t);
        }
    }

    //Initialize variable and check for error in the architecture of the model
    void Compile()
    {



        for_each_tuple<decltype(layers)>(layers);

        //Initialize the matrix which holds the values of the cost derivative.
        costDerivative = new LMAT<OutputShape>();

        /*
        auto* inputLayer = (Layer<InputLayer<InputShape>>*) std::get<0>(layers);
        if (inputLayer == nullptr)
            throw std::invalid_argument("First layer must be an input layer");
        */
        compiled = true;
    }

    //Load network from a file
    //TODO: Implement the load with templates
    static Network* Load(const std::string& filename)
    {
      /*
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
       */

    }

    //Save network to a file

    void Save(const std::string& filename)
    {
      /*
        std::ofstream writer;
        writer.open(filename, std::ios::binary | std::ios::trunc);

        //TODO: Implement the save of the loss with templates
        //loss->Save(writer);
        //First save the number of layers
        writer.write(reinterpret_cast<char*>(&layersCount), sizeof(int));

        for (int i = 0; i < layersCount; i++)
            Layers[i]->Save(writer);

        writer.close();
       */
    }

    template<int I>
    auto* GetLayer()
    {
        return &std::get<I>(layers);
    }


private:

    template<int I = 0>
    void UpdateWeightsRecu(double learningRate, int batchSize) {
        if constexpr (I < layersCount) {
            std::get<I>(layers).UpdateWeights(learningRate, batchSize);
            UpdateWeightsRecu<I + 1>(learningRate, batchSize);
        }
    }

    void UpdateWeights(double learningRate, int batchSize)
    {
        UpdateWeightsRecu(learningRate,batchSize);
        // for (int i = 1; i < layersCount; i++)
        //     layers[i]->UpdateWeights(learningRate, batchSize);
    }

template<typename InputShapeL = InputShape, typename OutputShapeL = OutputShape, int I = 0>
auto FeedForwardLearnRecu(const LMAT<InputShapeL>* input, const LMAT<OutputShapeL>* desiredOutput)
{
    if constexpr (I == layersCount)
    {
        return Loss::Cost(std::get<layersCount - 1>(layers).getResult(), desiredOutput);
    }
    else
    {
        using CurrentLayerShape = typename LayersType_At<I>::Shape;
        //using NextLayerShape = typename LayersType_At<I + 1>::Shape;

        auto layerOutput = std::get<I>(layers).FeedForward(input);

        return FeedForwardLearnRecu<CurrentLayerShape, OutputShapeL, I + 1>(layerOutput, desiredOutput);
    }
}
    //Compute values and loss
    double FeedForward(LMAT<InputShape>* input, LMAT<OutputShape>* desiredOutput)
    {
        return FeedForwardLearnRecu(input,desiredOutput);
        // output = input;
        // for (int i = 0; i < layersCount; i++)
        // {
        //     //std::cout << "feedforward : " << i << "\n";
        //     output = layers[i]->FeedForward(output);
        // }
        // //std::cout << *output;
        // return Loss::Cost(output, desiredOutput);
    }

    //The output of the network
    const LMAT<OutputShape>* output = nullptr;

    //The partial derivative of the cost
    LMAT<OutputShape>* costDerivative = nullptr;

    std::tuple<Layers...> layers;


    //Is Compiled ?
    bool compiled = false;


    Opti opti;

public:
    static inline int ctr = 0;
};






template<typename Loss,typename... Types>
Network<Loss,Types...>::Network(Network* network)
{
    for (int i = 0; i < layersCount; i++)
        this->layers[i] = network->layers[i]->Clone();

}

/*
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
*/
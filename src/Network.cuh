#pragma once

#include <iostream>
#include "Layer.cuh"
#include "Matrix.cuh"
#include "Loss.cuh"
#include "Optimizers.cuh"
#include "DataLoader.cuh"
#include <mutex>

class Network
{
public:
    Network();

    ~Network();

    explicit Network(Network* network);

    //Add a layer to the network
    void AddLayer(Layer* layer);

    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(MAT* input, MAT* output);

    void Learn(int epochs, double learningRate, MAT** inputs, MAT** outputs, int dataLength);

    //Compute a value threw the neural network
    MAT* Process(MAT* input);

    //Compute values through the neural network
    const MAT* FeedForward(MAT* input);


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
    MAT* costDerivative;

    Layer** Layers;

    //Loss function
    Loss* loss;

    //Is Compiled ?
    bool compiled = false;

    //The number of layers
    int layersCount = 0;

    Opti opti;

public:
    static inline int ctr = 0;
};


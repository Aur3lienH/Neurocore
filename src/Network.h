#pragma once

#include <iostream>
#include "Layer.h"
#include "Matrix.h"
#include "Loss.h"
#include "Optimizers.h"
#include "DataLoader.h"
#include <mutex>

class Network
{
public:
    Network();

    explicit Network(Network* network);

    //Add a layer to the network
    void AddLayer(Layer* layer);

    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(Matrix* input, Matrix* output);

    /// @brief Multithreading learning 
    /// @param epochs Number of times which the neural network will see the dataset
    /// @param learningRate 
    /// @param inputs The inputs of the dataset
    /// @param outputs The outputs of the dataset
    /// @param batchSize The number of turn before updating weights
    /// @param dataLength The size of the dataset
    /// @param threadNumber The number of thread used to train the model
    void Learn(int epochs, double learningRate, DataLoader* dataLoader, int batchSize, int threadNumber);

    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength);

    //Clear all delta from all layers (partial derivative)
    void ClearDelta();

    void PrintNetwork();

    //Compute a value threw the neural network
    Matrix* Process(Matrix* input);

    //Initialize variable and check for error in the architecture of the model
    void Compile(Opti opti = Opti::Constant, Loss* loss = nullptr);

    //Load network from a file
    static Network* Load(const std::string& filename);

    //Save network to a file
    void Save(const std::string& filename);

    //Compute values through the neural network
    const Matrix* FeedForward(Matrix* input);

private:
    void UpdateWeights(double learningRate, int batchSize);

    //Compute values and loss
    double FeedForward(Matrix* input, Matrix* desiredOutput);

    //The output of the network
    const Matrix* output = nullptr;

    Layer** Layers;

    //Loss function
    Loss* loss;

    //Is Compiled ?
    bool compiled = false;

    //The number of layers
    int layersCount = 0;

    //The partial derivative of the cost
    Matrix* costDerivative;

    Opti opti;
};


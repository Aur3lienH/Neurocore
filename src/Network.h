#pragma once
#include <iostream>
#include "Layer.h"
#include "Matrix.h"
#include "Loss.h"

class Network
{
public:
    Network();
    Network(Loss* loss);
    Network(Network* network);

    //Add a layer to the network
    void AddLayer(Layer* layer);

    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(Matrix* input,Matrix* output);

    /// @brief Multithreading learning 
    /// @param epochs Number of times which the neural network will see the dataset
    /// @param learningRate 
    /// @param inputs The inputs of the dataset
    /// @param outputs The outputs of the dataset
    /// @param batchSize The number of turn before updating weights
    /// @param dataLength The size of the dataset
    /// @param threadNumber The number of thread used to train the model
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int batchSize,int dataLength, int threadNumber);
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength);

    //Clear all delta from all layers (partial derivative)
    void ClearDelta();

    //Function to start a thread 
    static void* LearnThread(void* args);

    void PrintNetwork();

    //Compute a value threw the neural network
    Matrix* Process(Matrix* input);

    //Only SOFTMAX ! Test the accuracy of the model on a given testset
    double TestAccuracy(Matrix** inputs, Matrix** outputs, int dataLength);

    //Initialize variable and check for error in the architecture of the model
    void Compile(Loss* loss);
    void Compile();

    //Load network from a file
    static Network* Load(std::string filename);
    //Save network to a file
    void Save(std::string filename);

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
};


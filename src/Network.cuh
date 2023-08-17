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

#if USE_GPU
    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(Matrix_GPU* input, Matrix_GPU* output);

    void Learn(int epochs, double learningRate, Matrix_GPU** inputs, Matrix_GPU** outputs, int dataLength);

    //Compute a value threw the neural network
    Matrix_GPU* Process(Matrix_GPU* input);

    //Compute values through the neural network
    const Matrix_GPU* FeedForward(Matrix_GPU* input);
#else
    //Backpropagate threw the Network and store all the derivative
    double BackPropagate(Matrix* input, Matrix* output);

    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength);

    //Compute a value threw the neural network
    Matrix* Process(Matrix* input);

    //Compute values through the neural network
    const Matrix* FeedForward(Matrix* input);
#endif
    /// @brief Multithreading learning 
    /// @param epochs Number of times which the neural network will see the dataset
    /// @param learningRate 
    /// @param inputs The inputs of the dataset
    /// @param outputs The outputs of the dataset
    /// @param batchSize The number of turn before updating weights
    /// @param dataLength The size of the dataset
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

#if USE_GPU
    //Compute values and loss
    double FeedForward(Matrix_GPU* input, Matrix_GPU* desiredOutput);

    //The output of the network
    const Matrix_GPU* output = nullptr;

    //The partial derivative of the cost
    Matrix_GPU* costDerivative;
#else
    //Compute values and loss
    double FeedForward(Matrix* input, Matrix* desiredOutput);

    //The output of the network
    const Matrix* output = nullptr;

    //The partial derivative of the cost
    Matrix* costDerivative;
#endif

    Layer** Layers;

    //Loss function
    Loss* loss;

    //Is Compiled ?
    bool compiled = false;

    //The number of layers
    int layersCount = 0;

    Opti opti;
};


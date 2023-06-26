#pragma once
#include <iostream>
#include "Layer.h"
#include "Matrix.h"
#include "Loss.h"
#include "LastLayer.h"

class Network
{
public:
    Network();
    Network(Network* network);
    double BackPropagate(Matrix* input,Matrix* output);
    void AddLayer(Layer* layer);
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int batchSize,int dataLength, int threadNumber);
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength);
    void ClearDelta();
    static void* LearnThread(void* args);
    void PrintNetwork();
    Matrix* Process(Matrix* input);
    double TestAccuracy(Matrix** inputs, Matrix** outputs, int dataLength);
    void Compile();

    //Load network from a file
    static Network* Load(std::string filename);
    //Save network to a file
    void Save(std::string filename);
    
private:
    void UpdateWeights(double learningRate, int batchSize);
    Matrix* FeedForward(Matrix* input);
    double FeedForward(Matrix* input, Matrix* desiredOutput);
    Matrix* output = nullptr;
    Layer** Layers;
    LastLayer* lastLayer;
    bool compiled = false;
    int layersCount = 0;
};


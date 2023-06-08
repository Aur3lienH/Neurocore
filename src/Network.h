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
    Network(Network* network, Matrix** Delta, Matrix** DeltaBiases);
    double BackPropagate(Matrix* input,Matrix* output);
    void AddLayer(Layer* layer);
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int batchSize,int dataLength, int threadNumber);
    void Learn(int epochs, double learningRate, Matrix** inputs, Matrix** outputs, int dataLength);
    void ClearDelta();
    static void* LearnThread(void* args);
    void PrintNetwork();
    Matrix* FeedForward(Matrix* input);
    double FeedForward(Matrix* input, Matrix* desiredOutput);
    double TestAccuracy(Matrix** inputs, Matrix** outputs, int dataLength);
    void Compile();
    
private:
    void UpdateWeights(double learningRate, int batchSize);
    Matrix* output = nullptr;
    Layer** Layers;
    LastLayer* lastLayer;
    bool compiled = false;
    int layersCount = 0;
};


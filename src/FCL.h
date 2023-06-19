#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "Layer.h"
#include "./Tools/Serializer.h"

class FCL : public Layer
{
public:
    FCL(int NeuronsCount, Activation* activation);
    FCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaActivation);
    Matrix* FeedForward(const Matrix* input);
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights);
    void ClearDelta();
    void UpdateWeights(double learningRate, int batchSize);
    void UpdateWeights(double learningRate, int batchSize, Matrix* delta,Matrix* deltaActivation);
    void Compile(int previousNeuronsCount);
    Matrix* getDelta();
    Matrix* getDeltaBiases();
    Matrix* getResult() const;
    std::string getLayerTitle();
    virtual Layer* Clone(Matrix* delta, Matrix* deltaBiases);
    static FCL* Load(std::ifstream& ifstream);
    void SpecificSave(std::ofstream& filename);
protected:
    Matrix* Delta = nullptr;
    Matrix* Result = nullptr;
    Matrix* Weigths = nullptr;
    Matrix* Biases = nullptr;
    Matrix* DeltaBiases = nullptr;
    Activation* activation = nullptr;
    int NeuronsCount;
private:
    Matrix* newDelta = nullptr;


    Matrix* z = nullptr;
    Matrix* deltaActivation = nullptr;

    int previousNeuronsCount;


};


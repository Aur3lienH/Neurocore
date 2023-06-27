#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "Layer.h"
#include "./Tools/Serializer.h"
#include "LayerShape.h"

class FCL : public Layer
{
public:
    FCL(int NeuronsCount, Activation* activation);
    FCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaActivation);

    //Compute the input threw the layer
    Matrix* FeedForward(const Matrix* input) override;

    //Compute partial derivative (named delta)
    Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    //Clear partial derivative (named delta)
    void ClearDelta() override;

    //Update the current weights thanks to partial derivative (named delta)
    void UpdateWeights(double learningRate, int batchSize) override;

    //Add Delta from another identical layer
    void AddDeltaFrom(Layer* otherLayer) override;

    //Initialize variable and check for network architecture
    void Compile(LayerShape* previousLayer) override;

    //Getter for delta
    const Matrix* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const Matrix* getDeltaBiases();

    //Getter for the result of the layer
    const Matrix* getResult() const override;

    //Return information on the layer (neurons count)
    std::string getLayerTitle() override;

    //Clone layer
    Layer* Clone() override;

    static FCL* Load(std::ifstream& ifstream);

    void SpecificSave(std::ofstream& filename) override;
protected:
    //Partial derivative of the weights
    Matrix* Delta = nullptr;

    //Partial derivative of the biases
    Matrix* DeltaBiases = nullptr;

    //Results of the layer
    Matrix* Result = nullptr;

    Matrix* Weights = nullptr;
    Matrix* Biases = nullptr;
    
    //Activation function
    Activation* activation = nullptr;

    int NeuronsCount;
private:
    //Delta passed to the previous network in backpropagation
    Matrix* newDelta = nullptr;

    //Result before passing through the activation function
    Matrix* z = nullptr;

    //Delta from the activation function
    Matrix* deltaActivation = nullptr;

    //Neurons in the previous layer
    int previousNeuronsCount;


};


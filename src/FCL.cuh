#pragma once

#include "Matrix.cuh"
#include "Activation.cuh"
#include "Layer.cuh"
#include "./Tools/Serializer.h"
#include "LayerShape.cuh"

class FCL : public Layer
{
public:
    FCL(int NeuronsCount, Activation* activation);

    ~FCL() override;

    FCL(int NeuronsCount, Activation* activation, MAT* weights, MAT* bias, MAT* delta,
        MAT* deltaActivation);

    //Compute the input threw the layer
    MAT* FeedForward(const MAT* input) override;

    //Compute partial derivative (named delta)
    const MAT* BackPropagate(const MAT* delta, const MAT* lastWeights) override;

    //Getter for delta
    const MAT* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const MAT* getDeltaBiases();

    //Getter for the result of the layer
    [[nodiscard]] const MAT* getResult() const override;

    //Clear partial derivative (named delta)
    void ClearDelta() override;

    //Update the current weights thanks to partial derivative (named delta)
    void UpdateWeights(double learningRate, int batchSize) override;

    //Add Delta from another identical layer
    void AddDeltaFrom(Layer* otherLayer) override;

    //Initialize variable and check for network architecture
    void Compile(LayerShape* previousLayer) override;

    //Return information on the layer (neurons count)
    std::string getLayerTitle() override;

    //Clone layer
    Layer* Clone() override;

    static FCL* Load(std::ifstream& ifstream);

    void SpecificSave(std::ofstream& filename) override;

    void AverageGradients(int batchSize) override;

#if USE_GPU

    void Save(const std::string& folderPath, int n);

#else
    void Compare(const std::string& folderPath, int n);
#endif

protected:
//Partial derivative of the weights
    MAT* Delta = nullptr;

    //Partial derivative of the biases
    MAT* DeltaBiases = nullptr;

    //Results of the layer
    MAT* Result = nullptr;

    MAT* Weights = nullptr;
    MAT* Biases = nullptr;

    //Result before passing through the activation function
    MAT* z = nullptr;

    //Activation function
    Activation* activation = nullptr;

    int NeuronsCount;
private:
#if USE_GPU
    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
#else
    const Matrix* BackPropagateSSE2(const Matrix* delta, const Matrix* lastWeigths);

    const Matrix* BackPropagateAX2(const Matrix* delta, const Matrix* lastWeigths);
#endif

    //Delta passed to the previous network in backpropagation
    MAT* newDelta = nullptr;

    //Delta from the activation function
    MAT* deltaActivation = nullptr;
    //Neurons in the previous layer
    int previousNeuronsCount;

    float* buffer;


};


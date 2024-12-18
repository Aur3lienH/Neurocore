#pragma once

#include "matrix/Matrix.cuh"
#include "network/Activation.cuh"
#include "network/layers/Layer.cuh"
#include "tools/Serializer.h"
#include "network/LayerShape.cuh"

class FCL : public Layer<FCL>
{
public:
    FCL(int NeuronsCount, Activation* activation);

    ~FCL() override;

    FCL(int NeuronsCount, Activation* activation, MAT* weights, MAT* bias, MAT* delta,
        MAT* deltaActivation);

    //Compute the input threw the layer
    MAT* FeedForward(const MAT* input);

    //Compute partial derivative (named delta)
    const MAT* BackPropagate(const MAT* delta, const MAT* lastWeights);

    //Getter for delta
    const MAT* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const MAT* getDeltaBiases();

    //Getter for the result of the layer
    [[nodiscard]] const MAT* getResult() const;

    //Clear partial derivative (named delta)
    void ClearDelta();

    //Update the current weights thanks to partial derivative (named delta)
    void UpdateWeights(double learningRate, int batchSize);

    //Add Delta from another identical layer
    void AddDeltaFrom(Layer* otherLayer);

    //Initialize variable and check for network architecture
    void Compile(LayerShape* previousLayer);

    //Return information on the layer (neurons count)
    std::string getLayerTitle();

    //Clone layer
    Layer* Clone();

    static FCL* Load(std::ifstream& ifstream);

    void SpecificSave(std::ofstream& filename);

    void AverageGradients(int batchSize);

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
z
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


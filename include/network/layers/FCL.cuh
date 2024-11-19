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
    MAT* FeedForwardImpl(const MAT* input);

    //Compute partial derivative (named delta)
    const MAT* BackPropagateImpl(const MAT* delta, const MAT* lastWeights);

    //Getter for delta
    const MAT* getDeltaImpl();

    //Getter for deltaBiases (delta to update biases)
    const MAT* getDeltaBiasesImpl();

    //Getter for the result of the layer
    [[nodiscard]] const MAT* getResultImpl() const;

    //Clear partial derivative (named delta)
    void ClearDeltaImpl();

    //Update the current weights thanks to partial derivative (named delta)
    void UpdateWeightsImpl(double learningRate, int batchSize);

    //Add Delta from another identical layer
    void AddDeltaFromImpl(Layer* otherLayer);

    //Initialize variable and check for network architecture
    void CompileImpl(LayerShape* previousLayer);

    //Return information on the layer (neurons count)
    std::string getLayerTitleImpl();

    //Clone layer
    Layer* CloneImpl();

    static FCL* LoadImpl(std::ifstream& ifstream);

    void SpecificSaveImpl(std::ofstream& filename);

    void AverageGradientsImpl(int batchSize);

#if USE_GPU

    void SaveImpl(const std::string& folderPath, int n);

#else
    void CompareImpl(const std::string& folderPath, int n);
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


#pragma once

#include "Matrix.cuh"
#include "Activation.cuh"
#include "Layer.cuh"
#include "./Tools/Serializer.cuh"
#include "LayerShape.cuh"

class FCL : public Layer
{
public:
    FCL(int NeuronsCount, Activation* activation);

    ~FCL() override;

#if USE_GPU
    FCL(int NeuronsCount, Activation* activation, Matrix_GPU* weights, Matrix_GPU* bias, Matrix_GPU* delta,
        Matrix_GPU* deltaActivation);

    //Compute the input threw the layer
    Matrix_GPU* FeedForward(const Matrix_GPU* input) override;

    //Compute partial derivative (named delta)
    const Matrix_GPU* BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* lastWeights) override;

    //Getter for delta
    const Matrix_GPU* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const Matrix_GPU* getDeltaBiases();

    //Getter for the result of the layer
    [[nodiscard]] const Matrix_GPU* getResult() const override;
#else
    FCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta,
        Matrix* deltaActivation);

    //Compute the input threw the layer
    Matrix* FeedForward(const Matrix* input) override;

    //Compute partial derivative (named delta)
    const Matrix* BackPropagate(const Matrix* delta, const Matrix* lastWeights) override;

    //Getter for delta
    const Matrix* getDelta();

    //Getter for deltaBiases (delta to update biases)
    const Matrix* getDeltaBiases();

    //Getter for the result of the layer
    [[nodiscard]] const Matrix* getResult() const override;
#endif

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

protected:
#if USE_GPU
//Partial derivative of the weights
    Matrix_GPU* Delta = nullptr;

    //Partial derivative of the biases
    Matrix_GPU* DeltaBiases = nullptr;

    //Results of the layer
    Matrix_GPU* Result = nullptr;

    Matrix_GPU* Weights = nullptr;
    Matrix_GPU* Biases = nullptr;

    //Result before passing through the activation function
    Matrix_GPU* z = nullptr;
#else
    //Partial derivative of the weights
    Matrix* Delta = nullptr;

    //Partial derivative of the biases
    Matrix* DeltaBiases = nullptr;

    //Results of the layer
    Matrix* Result = nullptr;

    Matrix* Weights = nullptr;
    Matrix* Biases = nullptr;

    //Result before passing through the activation function
    Matrix* z = nullptr;
#endif

    //Activation function
    Activation* activation = nullptr;

    int NeuronsCount;
private:
#if USE_GPU
    //Delta passed to the previous network in backpropagation
    Matrix_GPU* newDelta = nullptr;

    //Delta from the activation function
    Matrix_GPU* deltaActivation = nullptr;
#else
    const Matrix* BackPropagateSSE2(const Matrix* delta, const Matrix* lastWeigths);

    const Matrix* BackPropagateAX2(const Matrix* delta, const Matrix* lastWeigths);

    //Delta passed to the previous network in backpropagation
    Matrix* newDelta = nullptr;

    //Delta from the activation function
    Matrix* deltaActivation = nullptr;
#endif
    //Neurons in the previous layer
    int previousNeuronsCount;

    float* buffer;


};


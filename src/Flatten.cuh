#pragma once

#include "Layer.cuh"

class Flatten : public Layer
{
public:
    Flatten();

#if USE_GPU
    const Matrix_GPU* FeedForward(const Matrix_GPU* input) override;

    const Matrix_GPU* BackPropagate(const Matrix_GPU* delta, const Matrix_GPU* pastActivation) override;

    [[nodiscard]] const Matrix_GPU* getResult() const override;
#else
    const Matrix* FeedForward(const Matrix* input) override;

    const Matrix* BackPropagate(const Matrix* delta, const Matrix* pastActivation) override;

    [[nodiscard]] const Matrix* getResult() const override;
#endif

    void ClearDelta() override;

    static Layer* Load(std::ifstream& reader);

    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;


    void Compile(LayerShape* previousOutput) override;

    std::string getLayerTitle() override;

    void SpecificSave(std::ofstream& writer) override;

    Layer* Clone() override;

    void AverageGradients(int batchSize) override;

private:
#if USE_GPU
    const Matrix_GPU* input;
#else
    const Matrix* input;
#endif
    int rows, cols, dims = 0;


};
#pragma once

#include "Layer.cuh"

class Flatten : public Layer
{
public:
    Flatten();

    const MAT* FeedForward(const MAT* _input) override;

    const MAT* BackPropagate(const MAT* delta, const MAT* pastActivation) override;

    [[nodiscard]] const MAT* getResult() const override;

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
    const MAT* input;
    int rows, cols, dims = 0;


};
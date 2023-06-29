#pragma once
#include "Layer.h"

class Flatten : public Layer
{
public:
    Flatten();
    const Matrix* FeedForward(const Matrix* input) override;
    const Matrix* BackPropagate(const Matrix* delta, const Matrix* pastActivation) override;
    void ClearDelta() override;

    static Layer* Load(std::ifstream& reader);


    void UpdateWeights(double learningRate, int batchSize) override;

    void AddDeltaFrom(Layer* layer) override;
    

    void Compile(LayerShape* previousOutput) override;

    std::string getLayerTitle() override;
    
    void SpecificSave(std::ofstream& writer);

    Layer* Clone();

    const Matrix* getResult() const override;

private:
    const Matrix* input;
    int rows, cols, dims = 0;


};
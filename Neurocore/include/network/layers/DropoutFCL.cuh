#pragma once

#include "matrix/Matrix.cuh"




template<typename LayerShape, double dropoutRate = .5>
class Dropout final
{
    [[nodiscard]] bool IsTraining() const
    { return isTraining; }

    void Save();

    void Drop();

public:
    Dropout(): gen(rd()), dist(1.0f - dropoutRate), keepprob(1.0f - dropoutRate)
    {
        mask = new bool[LayerShape::x * LayerShape::y * LayerShape::z];
    }
    ~Dropout() {
        delete[] mask;
    }

    const LMAT<LayerShape>* FeedForward(const LMAT<LayerShape>* input)
    {
        for(size_t i = 0; i < input->GetMatrixSize(); i++)
        {
            if(dist(gen)) {
                input->data[i] /= keepprob;
                mask[i] = true;
            }
            else {
                input->data[i] = 0;
                mask[i] = false;
            }
		}
	    return input;
    }

    const LMAT<LayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<LayerShape>* lastWeights)
    {
        for(size_t i = 0; i < delta->GetMatrixSize(); i++)
        {
            if(!mask[i]) {
                delta->data[i] = 0;
            }
        }
        return delta;
    }

    const MAT<1>* getDelta() {
        return nullptr;
    }

    const MAT<1>* getDeltaBiases() {
        return nullptr;
    }

    void UpdateWeights(double learningRate, int batchSize) {

    }

    void AverageGradients(int batchSize) {

    }

    void Compile() {

    }

private:
    std::random_device rd;
    std::mt19937 gen;
    std::bernoulli_distribution dist;
    bool isTraining = true;
    bool* mask;
    float keepprob;
};
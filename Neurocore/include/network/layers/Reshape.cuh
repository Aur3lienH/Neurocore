#pragma once

#include "Layer.cuh"

template<typename LayerShape, typename PrevLayerShape>
class Reshape final {
public:
    Reshape() {

    }

    const LMAT<LayerShape>* FeedForward(const LMAT<PrevLayerShape>* _input)
    {
        return output;
    }

    const LMAT<PrevLayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<PrevLayerShape>* pastActivation)
    {
        return newDelta;
    }

    [[nodiscard]] const LMAT<LayerShape>* getResult() const {
        return output;
    }

    void ClearDelta(){
    }

    //static Layer<Flatten>* Load(std::ifstream& reader);

    void UpdateWeights(double learningRate, int batchSize)
    {}

    void AddDeltaFrom(Layer<Reshape>* layer)
    {}

    template<int x, int y, int z, int size>
    void Compile()
    {}

    std::string getLayerTitle()
    {
        std::string buffer;
        buffer += "Layer : Reshape\n";
        buffer += "Output GetSize : " + std::to_string(LayerShape::x) + "\n";
        return buffer;
    }

    //void SpecificSave(std::ofstream& writer);

    //Layer<Flatten>* Clone();

    void AverageGradients(int batchSize) {

    }

private:
    const LMAT<LayerShape>* output;
    const LMAT<PrevLayerShape>* newDelta;
    int rows, cols, dims = 0;


};

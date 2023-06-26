#include "LastLayer.h"
#include "Activation.h"
#include "Loss.h"

LastLayer::LastLayer(int NeuronsCount, Activation* activation, Loss* _loss) : FCL(NeuronsCount, activation)
{
    this->loss = _loss;
    LayerID = 2;
}
LastLayer::LastLayer(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta, Matrix* deltaBiases, Loss* _loss) : FCL(NeuronsCount, activation, weights, bias, delta, deltaBiases)
{
    this->loss = _loss;
    LayerID = 2;
}

void LastLayer::ClearDelta()
{
    FCL::ClearDelta();
}

double LastLayer::getLossError()
{
    return lossError;
}

double LastLayer::FeedForward(const Matrix* input, const Matrix* desiredOutput)
{
    Matrix* output = FCL::FeedForward(input);
    return loss->Cost(output, desiredOutput);
    
}

Matrix* LastLayer::BackPropagate(const Matrix* desiredOutput, const Matrix* lastWeights)
{
    loss->CostDerivative(Result, desiredOutput, Result);
    return FCL::BackPropagate(Result, lastWeights);
}




Layer* LastLayer::Clone()
{
    return new LastLayer(NeuronsCount, activation, Weigths, Biases, nullptr,nullptr, loss);
}

Matrix* LastLayer::getDelta()
{
    return Delta;
}

Matrix* LastLayer::getDeltaBiases()
{
    return DeltaBiases;
}


void LastLayer::SpecificSave(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&NeuronsCount),sizeof(int));
    activation->Save(writer);
    Weigths->Save(writer);
    Biases->Save(writer);
    loss->Save(writer);
}

LastLayer* LastLayer::Load(std::ifstream& reader)
{
    int neuronsCount;
    reader.read(reinterpret_cast<char*>(&neuronsCount),sizeof(int));
    Activation* activation = Activation::Read(reader);
    Matrix* weights = Matrix::Read(reader);
    Matrix* biases = Matrix::Read(reader);
    Loss* loss = Loss::Read(reader);
    LastLayer* temp = new LastLayer(neuronsCount,activation,weights,biases,nullptr,nullptr,loss);
    return temp;
}


//
// Created by mat on 10/07/23.
//

#include "DropoutFCL.cuh"
#include "InitFunc.cuh"

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate) : FCL(NeuronsCount, activation),
                                                                                       dropoutRate(dropoutRate)
{

}

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, MAT* weights, MAT* bias, MAT* delta,
                       MAT* deltaActivation, double dropoutRate) : FCL(NeuronsCount, activation, weights, bias,
                                                                       delta, deltaActivation),
                                                                   dropoutRate(dropoutRate)
{

}

void DropoutFCL::Save()
{
    // Avoid memory leaks if function is called multiple times
    delete savedWeights;

    savedWeights = Weights->Copy();
}

void DropoutFCL::Compile(LayerShape* previousLayer)
{
    FCL::Compile(previousLayer);
    Save();
    Drop();
}

void DropoutFCL::SetIsTraining(const bool isTraining_)
{
#if USE_GPU
    throw std::runtime_error("DropoutFCL::SetIsTraining not implemented for GPU");
#else
    isTraining = isTraining_;
    if (isTraining)
    {
        Save();
        Drop();
    }
    else
    {
        Matrix* wtemp = savedWeights->Copy();
        Save();
        const double scale = 1.0 / (1.0 - dropoutRate);

        for (int i = 0; i < Weights->GetSize(); ++i)
        {
            if ((*Weights)[i] == 0)
            {
                (*Weights)[i] = (*wtemp)[i];
            }
            else
            {
                (*Weights)[i] /= scale;
            }
        }
    }
#endif
}

void DropoutFCL::Drop()
{
#if USE_GPU
    throw std::runtime_error("DropoutFCL::Drop not implemented for GPU");
#else
    const double scale = 1.0 / (1.0 - dropoutRate);

    // Zero out columns of droppedWeights and droppedBiases and scale the rest
    for (int i = 0; i < Weights->GetCols(); i++)
    {
        if (rand() / ((double) RAND_MAX) < dropoutRate)
        {
            for (int j = 0; j < Weights->GetRows(); j++)
            {
                (*Weights)(j, i) = 0;
            }
        }
        else
        {
            for (int j = 0; j < Weights->GetRows(); j++)
            {
                (*Weights)(j, i) *= scale;
            }
        }
    }
#endif
}
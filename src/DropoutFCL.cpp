//
// Created by mat on 10/07/23.
//

#include "DropoutFCL.h"
#include "InitFunc.h"

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, double dropoutRate) : FCL(NeuronsCount, activation),
                                                                                       dropoutRate(dropoutRate)
{

}

DropoutFCL::DropoutFCL(int NeuronsCount, Activation* activation, Matrix* weights, Matrix* bias, Matrix* delta,
                       Matrix* deltaActivation, double dropoutRate) : FCL(NeuronsCount, activation, weights, bias,
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

        for (int i = 0; i < Weights->size(); ++i)
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
}

void DropoutFCL::Drop()
{
    const double scale = 1.0 / (1.0 - dropoutRate);

    // Zero out columns of droppedWeights and droppedBiases and scale the rest
    for (int i = 0; i < Weights->getCols(); i++)
    {
        if (rand() / ((double) RAND_MAX) < dropoutRate)
        {
            for (int j = 0; j < Weights->getRows(); j++)
            {
                (*Weights)(j, i) = 0;
            }
        }
        else
        {
            for (int j = 0; j < Weights->getRows(); j++)
            {
                (*Weights)(j, i) *= scale;
            }
        }
    }
}
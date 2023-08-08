#include "InitFunc.h"
#include <cmath>

void XavierInit(const int inputSize, Matrix* weights)
{
    float upper = 1.0 / sqrt((float) inputSize);
    float lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
    }
};


void NormalizedXavierInit(const int inputSize, const int outputSize, Matrix* weights)
{
    float upper = (sqrt(6.0) / sqrt((float) inputSize + (float) outputSize));
    float lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
    }
};

void HeInit(const int inputSize, Matrix* weights)
{
    float range = sqrt(2.0 / (float) inputSize);

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = (rand() / ((float) RAND_MAX) - 0.5) * 2 * range;
    }

};


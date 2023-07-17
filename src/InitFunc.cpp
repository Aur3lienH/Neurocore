#include "InitFunc.h"
#include "math.h"

void XavierInit(int inputSize, Matrix* weights)
{
    double upper = 1.0/ sqrt((double)inputSize);
    double lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((double)RAND_MAX) * (upper - (lower)));
    }
};


void NormalizedXavierInit(int inputSize,int outputSize, Matrix* weights)
{
    double upper = (sqrt(6.0)/ sqrt((double)inputSize + (double)outputSize));
    double lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((double)RAND_MAX) * (upper - (lower)));
    }
};

void HeInit(int inputSize, Matrix* weights)
{
    double range = sqrt(2.0 / (double)inputSize);

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = (rand()/((double)RAND_MAX) -0.5) * 2 * range;
    }
    
};


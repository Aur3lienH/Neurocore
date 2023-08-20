#include "InitFunc.h"
#include <cmath>
#include <random>
#include <iostream>


std::mt19937 WeightsInit::rng = std::mt19937(std::random_device{}());


void WeightsInit::XavierInit(const int inputSize, Matrix* weights)
{
    float upper = 1.0 / sqrt((float) inputSize);
    float lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
    }
};


void WeightsInit::NormalizedXavierInit(const int inputSize, const int outputSize, Matrix* weights)
{
    float upper = (sqrt(6.0) / sqrt((float) inputSize + (float) outputSize));
    float lower = -upper;

    for (int i = 0; i < weights->size(); i++)
    {
        weights[0][i] = lower + (rand() / ((float) RAND_MAX) * (upper - (lower)));
    }
};

void WeightsInit::HeUniform(const int inputSize, Matrix* weights)
{
    double limit = std::sqrt(6.0 / inputSize);
    
    std::uniform_real_distribution<double> distribution(-limit, limit);

    for (int i = 0; i < weights->size(); ++i) {
        (*weights)[i] = distribution(rng);
    }
};


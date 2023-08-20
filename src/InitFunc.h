#pragma once
#include "Matrix.h"
#include <random>
#include <iostream>


//Class in which there are functions to init weigths
class WeightsInit
{
public:
    static void XavierInit(int inputSize, Matrix* weights);

    static void NormalizedXavierInit(int inputSize,int outputSize, Matrix* weights);

    static void HeUniform(int inputSize, Matrix* weights);
private:
    static std::mt19937 rng;
};



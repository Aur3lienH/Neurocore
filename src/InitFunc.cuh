#pragma once

#include "Matrix.cuh"
#include <random>
#include <iostream>

//Class in which there are functions to init weights
class WeightsInit
{
public:
    static void XavierInit(int inputSize, MAT* weights);

    static void NormalizedXavierInit(int inputSize, int outputSize, MAT* weights);

    static void HeUniform(int inputSize, MAT* weights);

private:
    static std::mt19937 rng;
};
#pragma once

#include "Matrix.cuh"


void XavierInit(int inputSize, MAT* weights);

void NormalizedXavierInit(int inputSize, int outputSize, MAT* weights);

void HeInit(int inputSize, MAT* weights);

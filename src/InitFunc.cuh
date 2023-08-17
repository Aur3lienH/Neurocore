#pragma once
#include "Matrix.cuh"


void XavierInit(int inputSize, Matrix* weights);

void NormalizedXavierInit(int inputSize,int outputSize, Matrix* weights);

void HeInit(int inputSize, Matrix* weights);

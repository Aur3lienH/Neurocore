#include "Layer.h"
#include "Matrix.h"


Layer::Layer(int* NeuronsCount, int NeuronsCountSize)
{
    this->NeuronsCount = NeuronsCount;
    this->NeuronsCountSize = NeuronsCountSize;
}

int Layer::getNeuronsCount(int index)
{
    return NeuronsCount[index];
}




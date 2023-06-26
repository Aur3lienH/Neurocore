#include "PoolingLayer.h"


void PoolingLayer::ClearDelta()
{
    delta->Zero();
}


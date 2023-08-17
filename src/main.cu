#include <iostream>
#include "Matrix.cuh"
#include "Network.cuh"
#include "Layer.cuh"
#include "FCL.cuh"
#include "Activation.cuh"
#include "InputLayer.cuh"
#include "Loss.cuh"
#include "Examples/Mnist.cuh"
#include "Examples/Tests.cuh"
#include <bits/stdc++.h>
#include "./Examples/Quickdraw.cuh"
#include "cudnn.h"


int main()
{
    Mnist1();
    //QuickDraw2(10000);
    return 0;
    //LoadAndTest("./Models/MNIST_11.net",true);
}
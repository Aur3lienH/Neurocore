#include <iostream>
#include "Matrix.h"
#include "Network.h"
#include "Layer.h"
#include "FCL.h"
#include "Activation.h"
#include "InputLayer.h"
#include "Loss.h"
#include "Examples/Mnist.h"
#include "Examples/Tests.h"
#include <bits/stdc++.h>
#include "./Examples/Quickdraw.h"


int main()
{
    Mnist2();
    //QuickDraw2(10000);
    return 0;
    FashionMnist2();
    //LoadAndTest("./Models/MNIST_11.net",true);
}
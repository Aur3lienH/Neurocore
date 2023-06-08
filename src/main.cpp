#include <iostream>
#include "Matrix.h"
#include "Network.h"
#include "Layer.h"
#include "FCL.h"
#include "Activation.h"
#include "LastLayer.h"
#include "InputLayer.h"
#include "Loss.h"
#include "Examples/Mnist.h"
#include <bits/stdc++.h>


int main()
{
    time_t start, end;
    time(&start);



    Network network = Network();
    network.AddLayer(new InputLayer(2));
    network.AddLayer(new FCL(10000, new Sigmoid()));
    network.AddLayer(new LastLayer(2, new Softmax(), new CrossEntropy()));

    network.Compile();

    network.PrintNetwork();


    Matrix** input = new Matrix*[2];
    Matrix** output = new Matrix*[2];
    input[0] = new Matrix(2,1,new double[2]{1,0});
    input[1] = new Matrix(2,1,new double[2]{0,1});
    output[0] = new Matrix(2,1,new double[2]{1,0});
    output[1] = new Matrix(2,1,new double[2]{0,1});
    std::cout << *network.FeedForward(input[0]) << std::endl;
    std::cout << *network.FeedForward(input[1]) << std::endl;
    network.Learn(10000,0.1,input,output,2,5,1);
    std::cout << *input[0] << std::endl;
    std::cout << *network.FeedForward(input[0]) << std::endl;
    std::cout << *network.FeedForward(input[1]) << std::endl;

    time(&end);
    double time_taken = double(end) - double(start);
    std::cout << "Time taken by program is : " << std::fixed
         << time_taken << std::setprecision(5) << " sec " << std::endl;
}
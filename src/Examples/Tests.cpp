#include "Tests.h"
#include <iostream>
#include "../Network.h"
#include "../InputLayer.h"
#include "../Activation.h"
#include <limits>
#include <iomanip>


void Tests::BasicNetwork1()
{
    time_t start, end;
    time(&start);



    Network network = Network();
    network.AddLayer(new InputLayer(2));
    network.AddLayer(new FCL(5, new Sigmoid()));
    network.AddLayer(new FCL(5, new Sigmoid()));
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
    network.Learn(200,0.1,input,output,1,2,1);
    std::cout << *input[0] << std::endl;
    std::cout << *network.FeedForward(input[0]) << std::endl;
    std::cout << *network.FeedForward(input[1]) << std::endl;

    time(&end);
    double time_taken = double(end) - double(start);
    std::cout << "Time taken by program is : " << std::fixed
         << time_taken << std::setprecision(5) << " sec " << std::endl;
}


void Tests::BasicNetwork2()
{
    Network network = Network();
    network.AddLayer(new InputLayer(2));
    network.AddLayer(new FCL(2, new Sigmoid()));
    network.AddLayer(new LastLayer(2, new Softmax(), new CrossEntropy()));

    network.Compile();


    Matrix** input = new Matrix*[2];
    Matrix** output = new Matrix*[2];
    input[0] = new Matrix(2,1,new double[2]{1,0});
    input[1] = new Matrix(2,1,new double[2]{0,1});
    output[0] = new Matrix(2,1,new double[2]{1,0});
    output[1] = new Matrix(2,1,new double[2]{0,1});

    std::cout << *network.FeedForward(input[0]);
    std::cout << '\n';

    network.Learn(1,1,input,output,1,2,1);
    
}
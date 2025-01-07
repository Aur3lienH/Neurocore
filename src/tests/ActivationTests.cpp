#include "tests/ActivationTests.h"
#include "network/activation/ReLU.h"
#include "network/activation/LeakyReLU.h"
#include "network/activation/Tanh.h"
#include "network/activation/Softmax.h"
#include "network/activation/Activation.cuh"


bool ActivationTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));

    bool* array = new bool[functions.size()];


    for (int i = 0; i < functions.size(); i++)
    {
        bool (*func)(void) = (bool (*)(void))(std::get<0>(functions[i]));
        bool res = func();
        if(res)
        {
            array[i] = true;
        }
        else
        {
            array[i] = false;
            res = false;
        }
    }

    //system("clear");
    for (int i = 0; i < functions.size(); i++) {
        if(array[i])
        {
            std::cout << "\033[1;32m[SUCCEED]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
        else
        {
            std::cout << "\033[1;31m[FAIL]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
    }
    free(array);
    return res;
}

//Test for
//  ReLU::FeedForward(Matrix*,Matrix*)
//  ReLU::Derivative(Matrix*,Matrix*)
/*
bool ActivationTests::TestReLUFunction()
{
    typedef Activation<ReLU<5,1,1>> ReLU;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    ReLU::FeedForward(&input, &output);
    if(output[0] != 0 || output[1] != 2 || output[2] != 0 || output[3] != 4 || output[4] != 0)
    {
        return false;
    }

    return true;
}

*/


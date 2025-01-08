#include "tests/NetworkTests.h"
#include "network/Network.h"
#include "network/layers/FCL.cuh"
#include "network/activation/ReLU.h"
#include "network/layers/InputLayer.cuh"
#include "network/loss/MSE.cuh"
#include "network/loss/Loss.h"
#include <iostream>
#include <functional>


bool NetworkTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    functions.push_back(std::tuple((void*)BasicFFN,std::string("Basic FFN")));

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
            std::cout << "  \033[1;32m[SUCCEED]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
        else
        {
            std::cout << "  \033[1;31m[FAIL]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
    }
    free(array);
    return res;
}


bool NetworkTests::BasicFFN()
{
    Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>>
    > neuralnet;
    //neuralnet.Compile();

    return true;
}





#include "tests/LayerTests.h"
#include "network/layers/FCL.cuh"
#include "network/layers/InputLayer.cuh"
#include <functional>
#include <iostream>


bool LayerTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));
    functions.push_back(std::tuple((void*)TestInputLayer, std::string("Layers")));

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

bool LayerTests::TestFCLLayer()
{
    /*FCL<Activation<ReLU<5,1,1>>,LayerShape<2>,LayerShape<5>> fcl;
    fcl.Compile();
    Matrix<2> input({1,2});
    Matrix<5>* out = fcl.FeedForward(&input);
    if(out->GetCols() != 5 || out->GetRows() != 1)
    {
        return false;
    }
    std::cout << out;
    */
    return true;
}

bool LayerTests::TestInputLayer()
{
    InputLayer<LayerShape<5>> inputlayer;
    inputlayer.Compile();

    const Matrix<5> input({1,1,1,1,1});
    const Matrix<5>* out = inputlayer.FeedForward(&input);
    return out->GetRows() == 5 && out->GetCols() == 1;
}

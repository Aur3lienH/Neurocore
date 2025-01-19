#include "tests/LayerTests.h"
#include "network/layers/FCL.cuh"
#include "network/layers/InputLayer.cuh"
#include <functional>
#include <iostream>
#include "network/layers/ConvLayer.cuh"

bool LayerTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));
    functions.emplace_back((void*)TestInputLayer, std::string("Input Layer"));
    functions.emplace_back((void*)TestFCLLayer, std::string("FCL layer"));
    functions.emplace_back((void*)TestCNNLayer, std::string("CNN layer"));

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

bool LayerTests::TestFCLLayer()
{

    FCL<Activation<ReLU<5,2,1,1>>,LayerShape<2>,LayerShape<5>> fcl;
    fcl.Compile();
    Matrix<2> input({1,2});
    Matrix<5>* out = fcl.FeedForward(&input);
    if(out->GetRows() != 5 || out->GetCols() != 1)
    {
        return false;
    }

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

bool LayerTests::TestCNNLayer()
{
    ConvLayer<Activation<ReLU<1,3>>, LayerShape<3,3>, LayerShape<1,1>, LayerShape<3,3>, Constant<1.0>, true> cnn;
    cnn.Compile();
    Matrix<3,3> filters({0,0,0,0,1,0,0,0,0});
    cnn.SetWeights(&filters);
    Matrix<1,1> biases(0.);
    cnn.SetBiases(&biases);
    Matrix<3,3> input({1,1,1,1,2,1,1,1,1,1});
    Matrix<1,1>* out = cnn.FeedForward(&input);
    Matrix<1,1> delta(1);
    Matrix<3,3>* bout = cnn.BackPropagate(&delta, &input);

    return out->data[0] == 2;
}
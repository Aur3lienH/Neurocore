#include "tests/LossTests.h"

#include <memory>

#include "network/loss/MSE.cuh"
#include "network/loss/CrossEntropy.cuh"
#include "network/loss/Loss.h"

bool LossTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    functions.emplace_back((void*)LossTests::TestMSELoss,std::string("MSE Loss"));
    functions.emplace_back((void*)LossTests::TestCrossEntropyLoss,std::string("Cross Entropy Loss"));

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
    delete[] array;
    return res;
}

//Test for
//  MSE::Cost(Matrix*,Matrix*)
//  MSE::CostDerivative(Matrix*,Matrix*,Matrix*)
bool LossTests::TestMSELoss()
{

    MAT<2,1,1> input({1,2});
    MAT<2,1,1> desiredOutput({3,4});
    double cost = Loss<MSE<2,1,1>>::Cost(&input, &desiredOutput);
    if(cost - 2.5 >= 1e-6)
    {
        return false;
    }

    MAT<2,1,1> output;
    Loss<MSE<2,1,1>>::CostDerivative(&input, &desiredOutput, &output);
    if(output.get(0) + 2 >= 1e-6 || output.get(1) + 2 >= 1e-6)
    {
        return false;
    }
    return true;
}

//Test for
//  CrossEntropy::Cost(Matrix*,Matrix*)
//  CrossEntropy::CostDerivative(Matrix*,Matrix*,Matrix*)
bool LossTests::TestCrossEntropyLoss()
{
    MAT<2,1,1> input({0.5,0.5});
    MAT<2,1,1> desiredOutput({1,0});
    double cost = Loss<CrossEntropy<2,1,1>>::Cost(&input, &desiredOutput);
    if(std::abs(cost - 0.693147) >= 1e-6)
    {
        return false;
    }

    MAT<2,1,1> output;
    Loss<CrossEntropy<2,1,1>>::CostDerivative(&input, &desiredOutput, &output);
    if(std::abs(output.get(0) + 0.5) >= 1e-6 || std::abs(output.get(1) - 0.5) >= 1e-6)
    {
        return false;
    }
    return true;
}
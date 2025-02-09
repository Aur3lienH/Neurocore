#include "tests/Tests.h"



#include <iostream>
//#include "tests/MatrixTests.h"
//#include "tests/LayerTests.h"
#include "tests/ActivationTests.h"
//#include "tests/LossTests.h"
//#include "tests/NetworkTests.h"
#include <limits>
#include <tuple>
#include <vector>
#include <iomanip>
#include <chrono>
#include <functional>


void Tests::ExecuteTests()
{
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));
    //functions.emplace_back((void*)MatrixTests::ExecuteTests,std::string("MATRIX TESTS"));
    //functions.emplace_back((void*)LayerTests::ExecuteTests,std::string("LAYER TESTS"));
    functions.emplace_back((void*)ActivationTests::ExecuteTests,std::string("ACTIVATION TESTS"));
    ///functions.emplace_back((void*)LossTests::ExecuteTests,std::string("LOSS TESTS"));
    ///functions.emplace_back((void*)NetworkTests::ExecuteTests,std::string("NETWORK TESTS"));


    
    for (int i = 0; i < functions.size(); i++)
    {
        bool (*func)(void) = (bool (*)(void))(std::get<0>(functions[i]));
        bool res = func();
        if(res)
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


}


void Tests::ExecuteSpeedTests()
{
    std::vector<std::tuple<void*, std::string>> functions;

    for (int i = 0; i < functions.size(); i++)
    {
        void (*func)(void) = (void (*)(void))(std::get<0>(functions[i]));

        auto start = std::chrono::high_resolution_clock::now();

        func();

        auto end = std::chrono::high_resolution_clock::now();

        double delta = (end - start).count();

        std::cout << std::get<1>(functions[i]) << "\n";
        std::cout << "duration : " << delta << " \n";
    }
    
}



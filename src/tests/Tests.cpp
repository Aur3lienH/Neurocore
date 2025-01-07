#include "tests/Tests.h"



#include <iostream>
#include "tests/MatrixTests.h"
#include "tests/LayerTests.h"
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
    functions.emplace_back((void*)MatrixTests::ExecuteTests,std::string("MATRIX TESTS"));
    //functions.emplace_back((void*)LayerTests::ExecuteTests,std::string("LAYER TESTS"));

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
        }
    }

    //system("clear");
    for (int i = 0; i < functions.size(); i++)
    {
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



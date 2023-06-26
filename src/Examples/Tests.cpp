#include "Tests.h"
#include <iostream>
#include "../Network.h"
#include "../InputLayer.h"
#include "../Activation.h"
#include "../ConvLayer.h"
#include <limits>
#include <tuple>
#include <iomanip>


void Tests::ExecuteTests()
{
    std::vector<std::tuple<void*,std::string>> functions;
    functions.push_back(std::make_tuple((void*)BasicNetwork1,"Single Thread"));
    //functions.push_back(std::make_tuple((void*)SaveNetwork1,"Save and Load Network"));
    functions.push_back(std::make_tuple((void*)CNNNetwork1,"Basic convolution test"));

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


bool Tests::BasicNetwork1()
{
    Network network = Network();
    network.AddLayer(new InputLayer(2));
    network.AddLayer(new FCL(5, new Sigmoid()));
    network.AddLayer(new FCL(5, new Sigmoid()));
    network.AddLayer(new LastLayer(2, new Softmax(), new CrossEntropy()));
    network.Compile();


    Matrix** input = new Matrix*[2];
    Matrix** output = new Matrix*[2];
    input[0] = new Matrix(2,1,new double[2]{1,0});
    input[1] = new Matrix(2,1,new double[2]{0,1});
    output[0] = new Matrix(2,1,new double[2]{1,0});
    output[1] = new Matrix(2,1,new double[2]{0,1});

    Matrix* res2 = network.Process(input[0]);
    Matrix* res3 = network.Process(input[1]);

    std::cout << *res2 << '\n';
    std::cout << *res3 << '\n';


    network.Learn(100000,0.01,input,output,1,2,1);

    Matrix* res0 = network.Process(input[0]);
    std::cout << res0 << '\n';
    Matrix* res1 = network.Process(input[1]);
    std::cout << res1 << '\n';
    if(Matrix::Distance(res0,output[0]) < 0.01f && Matrix::Distance(res1,output[1]) < 0.01)
    {
        return true;
    }
    return false;
}


bool Tests::SaveNetwork1()
{
    Network network = Network();
    network.AddLayer(new InputLayer(2));
    network.AddLayer(new FCL(5, new Sigmoid()));
    network.AddLayer(new FCL(5, new Sigmoid()));
    network.AddLayer(new LastLayer(2, new Softmax(), new CrossEntropy()));

    network.Compile();

    Matrix** input = new Matrix*[2];
    Matrix** output = new Matrix*[2];
    input[0] = new Matrix(2,1,new double[2]{1,0});
    input[1] = new Matrix(2,1,new double[2]{0,1});
    output[0] = new Matrix(2,1,new double[2]{1,0});
    output[1] = new Matrix(2,1,new double[2]{0,1});
    network.Learn(100000,0.1,input,output,1,2,1);
    network.Save("networkTest.net");
    Network* secondNet = Network::Load("networkTest.net");
    secondNet->Compile();

    Matrix* res0 = network.Process(input[0]);
    Matrix* res1 = network.Process(input[1]);
    if(Matrix::Distance(res0,output[0]) < 0.01f && Matrix::Distance(res1,output[1]) < 0.01)
    {
        return true;
    }
    return false;

}

bool Tests::CNNNetwork1()
{
    std::cout << "Network 1rst !\n";

    Matrix* filter = new Matrix(2,2,new double[4]{1,0,0,1});

    Network network = Network();
    network.AddLayer(new InputLayer(3,3,1));
    network.AddLayer(new ConvLayer(filter));

    network.Compile();


    Matrix* input1 = new Matrix(3,3,5);


    std::cout << "Before process \n";

    Matrix* out = network.Process(input1);

    std::cout << *out;

    if(out == input1)
    {
        return true;
    }
    
    return false;
}
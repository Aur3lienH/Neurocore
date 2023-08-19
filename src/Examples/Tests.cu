#include "Tests.cuh"
#include <iostream>
#include "../Network.cuh"
#include "../InputLayer.cuh"
#include "../Activation.cuh"
#include "../ConvLayer.cuh"
#include "../Flatten.cuh"
#include "../FCL.cuh"
#include <limits>
#include <tuple>
#include <iomanip>

/*
void Tests::ExecuteTests()
{
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::make_tuple((void*)BasicNetwork1,"Single Thread"));
    //functions.push_back(std::make_tuple((void*)SaveNetwork1,"Save and Load Network"));
    //functions.push_back(std::make_tuple((void*)CNNNetwork1,"Basic convolution test"));
    functions.push_back(std::make_tuple((void*)CNNSaveTest, "CNN save test"));

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
    for (int i = 0; i < functions.GetSize(); i++)
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
    network.AddLayer(new FCL(2, new Sigmoid()));
    network.Compile(Opti::Adam,new MSE());


    Matrix** input = new Matrix*[2];
    Matrix** output = new Matrix*[2];
    input[0] = new Matrix(2,1,new double[2]{1,0});
    input[1] = new Matrix(2,1,new double[2]{0,1});
    output[0] = new Matrix(2,1,new double[2]{1,0});
    output[1] = new Matrix(2,1,new double[2]{0,1});


    std::cout << "learning is beginning ! \n";
    network.Learn(10000,0.1,input,output,2);

    Matrix* res1 = network.Process(input[1]);
    std::cout << *res1 << '\n';

    Matrix* res0 = network.Process(input[0]);
    std::cout << *res0 << '\n';

    if(Matrix::Distance(res0,output[0]) < 0.1f && Matrix::Distance(res1,output[1]) < 0.1)
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
    network.AddLayer(new FCL(2, new Softmax()));

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
    network.AddLayer(new ConvLayer(new LayerShape(2,2,1),new ReLU()));

    network.Compile(Opti::Adam,new MSE());


    Matrix* input1 = new Matrix(3,3,5);


    std::cout << "Before process \n";

    const Matrix* out = network.FeedForward(input1);

    std::cout << *out;


    Matrix** inputs = new Matrix*[2];
    Matrix** outputs = new Matrix*[2];

    inputs[0] = new Matrix(3,3,new double[9]{1,1,1,0,0,0,0,0,0});
    inputs[1] = new Matrix(3,3,new double[9]{0,0,0,0,0,0,1,1,1});

    outputs[0] = new Matrix(2,2,20);
    outputs[1] = new Matrix(2,2,8);
    

    network.Learn(10000,0.1,inputs,outputs,2);

    std::cout << *network.FeedForward(inputs[0]);
    std::cout << *network.FeedForward(inputs[1]);

    if(out == input1)
    {
        return true;
    }
    
    return false;
}

bool Tests::CNNSaveTest()
{
    Network* network = new Network();
    network->AddLayer(new InputLayer(28,28,1));
    network->AddLayer(new ConvLayer(new LayerShape(2,2,32),new ReLU()));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(10, new Softmax()));

    network->Compile(Opti::Adam,new CrossEntropy());
    network->Save("test.net");

    Network* secondNetwork = Network::Load("test.net");

    Matrix* input = new Matrix(28,28);
    const Matrix* output = secondNetwork->FeedForward(input);

    std::cout << *output;
    return true;
}

*/
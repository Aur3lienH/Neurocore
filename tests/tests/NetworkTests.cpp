#include "tests/NetworkTests.h"
#include "network/Network.h"
#include "network/layers/FCL.cuh"
#include "network/activation/ReLU.h"
#include "network/layers/InputLayer.cuh"
#include "datasetsBehaviour/DataLoader.h"
#include "network/loss/MSE.cuh"
#include "network/loss/Loss.h"
#include <iostream>
#include <functional>

#include "network/layers/ConvLayer.cuh"
#include "network/layers/MaxPooling.cuh"
#include "network/layers/Reshape.cuh"


bool NetworkTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    functions.push_back(std::tuple((void*)BasicFFNFeedForward,std::string("Basic FFN")));
    functions.emplace_back((void*)DataLoaderTest,std::string("DataLoader"));
    functions.emplace_back((void*)BasicFFNLearn,std::string("Basic FFN Learn"));
    functions.emplace_back((void*)CNNMaxPoolTest, std::string("CNN MaxPool \"Mnist\""));
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
    delete[] array;
    return res;
}


bool NetworkTests::BasicFFNFeedForward()
{

    Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>,true>
    > neuralnet;
    neuralnet.Compile();
    MAT<1> input({1});
    auto* fcl = neuralnet.GetLayer<1>();
    fcl->SetWeights(new MAT<5>({{1,2,3,4,5}}));
    fcl->SetBiases(new MAT<5>({1,2,3,4,5}));
    const MAT<5>* output = neuralnet.FeedForward(&input);
    output->Print();

    // Expected values after computation
    const double expected[5] = {2, 4, 6, 8, 10};

    // Test each output value
    bool correct = true;
    for(int i = 0; i < 5; i++) {
        if(std::abs((*output)[i] - expected[i]) > 1e-6) {  // Using epsilon for float comparison
            std::cout << "Mismatch at position " << i << ": Expected "
                      << expected[i] << " but got " << (*output)[i] << std::endl;
            correct = false;
        }
    }
    return correct;
}

bool NetworkTests::BasicFFNLearn()
{


    Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>,true>
    > neuralnet;
    neuralnet.Compile();
    MAT<1> input({1});
    auto fcl = neuralnet.GetLayer<1>();
    fcl->SetWeights(new MAT<5>({{1,2,3,4,5}}));
    fcl->SetBiases(new MAT<5>({1,2,3,4,5}));
    Matrix<1> input2({1});
    Matrix<5> desiredOutput({-2,-4,-6,-8,-10});


    auto* dataset = new DataLoader<decltype(neuralnet)>(&input2,&desiredOutput,1);
    neuralnet.Learn(1, 0.01, dataset);
    auto* weights = fcl->GetWeights();
    auto* biases = fcl->GetBiases();
    const float values[5] = {0.96,1.92,2.88,3.84,4.80};
    for (size_t i = 0; i < 5; i++) {
        if (weights->data[i] - values[i] > 1e-6 || biases->data[i] - values[i] > 1e-6) {
            weights->Print();
            biases->Print();
            return false;
        }
    }

    return true;
}

bool NetworkTests::CNNMaxPoolTest()
{
    typedef Network<
        MSE<10,1,1>,
        InputLayer<LayerShape<28,28,1,1>>,
        ConvLayer<ReLU<26,28,26,16>,LayerShape<28,28,1,1>,LayerShape<26,26,16,1>,LayerShape<3,3,16,1>>,
        ConvLayer<ReLU<24,26,24,10>,LayerShape<26,26,16,1>,LayerShape<24,24,10,1>,LayerShape<3,3,10,1>>,
        MaxPoolLayer<LayerShape<12,12,10,1>,LayerShape<24,24,10,1>,2,2>,
        Reshape<LayerShape<1440,1,1,1>,LayerShape<12,12,10,1>>,
        FCL<ReLU<10,1440,1,1>,LayerShape<1440,1,1,1>,LayerShape<10,1,1,1>>
> NETWORK;
    NETWORK net;
    net.Compile();
    auto* input = Matrix<28,28,1>::Random();
    auto* output = new Matrix<10,1,1>({0,0,0,0,0,0,0,0,0,1});
    DataLoader<NETWORK> data = DataLoader<NETWORK>(input,output,1);
    net.Learn(1000,0.001,&data);
    double loss = net.FeedForward(input, output);
    return loss < 1e-3;
}

bool NetworkTests::DataLoaderTest() {

    typedef Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>,true>
    > net;
    auto dataset = DataLoader<net>(nullptr,nullptr,0);
    return true;

    return true;
}


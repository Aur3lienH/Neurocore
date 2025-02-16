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


bool NetworkTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    functions.push_back(std::tuple((void*)BasicFFNFeedForward,std::string("Basic FFN")));
    functions.emplace_back((void*)DataLoaderTest,std::string("DataLoader"));
    functions.emplace_back((void*)BasicFFNLearn,std::string("Basic FFN Learn"));
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


bool NetworkTests::BasicFFNFeedForward()
{

    Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>>
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
        if(std::abs(output->get(i) - expected[i]) > 1e-6) {  // Using epsilon for float comparison
            std::cout << "Mismatch at position " << i << ": Expected "
                      << expected[i] << " but got " << output->get(i) << std::endl;
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
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>>
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
    auto* weights = fcl->GetWeightsCPUCopy();
    auto* biases = fcl->GetBiasesCPUCopy();
    const float values[5] = {0.96,1.92,2.88,3.84,4.80};
    for (size_t i = 0; i < 5; i++) {
        if (weights->data[i] - values[i] > 1e-6 || biases->data[i] - values[i] > 1e-6) {
            weights->Print();
            biases->Print();
            return false;
        }
    }
    delete weights;
    delete biases;

    return true;
}

bool NetworkTests::DataLoaderTest() {

    typedef Network<
        Loss<MSE<5,1,1>>,
        InputLayer<LayerShape<1>>,
        FCL<ReLU<5,1,1,1>,LayerShape<1>,LayerShape<5>,Constant<0.01>>
    > net;
    auto dataset = DataLoader<net>(nullptr,nullptr,0);
    return true;

    return true;
}


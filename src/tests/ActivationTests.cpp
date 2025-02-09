#include "tests/ActivationTests.h"
#include "network/activation/ReLU.h"
#include "network/activation/LeakyReLU.h"
#include "network/activation/Tanh.h"
#include "network/activation/Softmax.h"
#include "network/activation/Activation.cuh"


bool ActivationTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    functions.emplace_back((void*)ActivationTests::TestReLU,std::string("ReLU"));
    functions.emplace_back((void*)ActivationTests::TestLeakyReLU,std::string("LeakyReLU"));
    functions.emplace_back((void*)ActivationTests::TestTanh,std::string("Tanh"));
    functions.emplace_back((void*)ActivationTests::TestSigmoid,std::string("Sigmoid"));
    functions.emplace_back((void*)ActivationTests::TestSoftmax,std::string("Softmax"));
    functions.emplace_back((void*)TestSigmoid,std::string("Sigmoid Prime"));

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

//Test for
//  ReLU::FeedForward(Matrix*,Matrix*)
//  ReLU::Derivative(Matrix*,Matrix*)
//  ReLU::InitWeights()
//  ReLU::InitBiases()
bool ActivationTests::TestReLU()
{
    //ReLU::FeedForward(Matrix*,Matrix*)
    typedef Activation<ReLU<5,2,1,1>> ReLU;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    ReLU::FeedForward(&input, &output);
    if(output[0] != 0 || output[1] != 2 || output[2] != 0 || output[3] != 4 || output[4] != 0) {
        return false;
    }

    //ReLU::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    ReLU::Derivative(&input2, &output2, nullptr, nullptr);
    if(output2[0] != 0 || output2[1] != 1 || output2[2] != 0 || output2[3] != 1 || output2[4] != 0) {
        return false;
    }

    //ReLU::InitWeights()
    MAT<5,2,1>* weights = ReLU::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        return false;
    }

    //ReLU::InitBiases()
    MAT<5,1,1>* biases = ReLU::InitBiases();
    if(biases->GetRows() != 5 || biases->GetCols() != 1) {
        return false;
    }


    return true;
}

//Test for
//  LeakyReLU::FeedForward(Matrix*,Matrix*)
//  LeakyReLU::Derivative(Matrix*,Matrix*)
//  LeakyReLU::InitWeights()
//  LeakyReLU::InitBiases()
bool ActivationTests::TestLeakyReLU()
{
    const float epsilon = 1e-6f;  // Define epsilon for float comparisons

    //LeakyReLU::FeedForward(Matrix*,Matrix*)
    typedef Activation<LeakyReLU<5,2>> LeakyReLU;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    LeakyReLU::FeedForward(&input, &output);

    if (std::abs(output[0] - (-0.01f)) > epsilon ||
        std::abs(output[1] - 2.0f) > epsilon ||
        std::abs(output[2] - (-0.03f)) > epsilon ||
        std::abs(output[3] - 4.0f) > epsilon ||
        std::abs(output[4] - (-0.05f)) > epsilon) {
        return false;
        }

    //LeakyReLU::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    LeakyReLU::Derivative(&input2, &output2, nullptr, nullptr);

    if (std::abs(output2[0] - 0.01f) > epsilon ||
        std::abs(output2[1] - 1.0f) > epsilon ||
        std::abs(output2[2] - 0.01f) > epsilon ||
        std::abs(output2[3] - 1.0f) > epsilon ||
        std::abs(output2[4] - 0.01f) > epsilon) {
        return false;
        }

    //LeakyReLU::InitWeights()
    MAT<5,2,1>* weights = LeakyReLU::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        delete weights;
        return false;
    }
    //LeakyReLU::InitBiases()
    MAT<5,1,1>* biases = LeakyReLU::InitBiases();
    if(biases->GetRows() != 5 || biases->GetCols() != 1) {
        delete weights;
        delete biases;
        return false;
    }

    delete weights;
    delete biases;
    return true;
}

//Test for
//  Tanh::FeedForward(Matrix*,Matrix*)
//  Tanh::Derivative(Matrix*,Matrix*)
//  Tanh::InitWeights()
bool ActivationTests::TestTanh()
{
    const float epsilon = 1e-6f;

    //Tanh::FeedForward(Matrix*,Matrix*)
    typedef Activation<Tanh<5,2,1,1>> Tanh;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    Tanh::FeedForward(&input, &output);

    if (std::abs(output[0] - (-0.761594f)) > epsilon ||
        std::abs(output[1] - 0.964028f) > epsilon ||
        std::abs(output[2] - (-0.995055f)) > epsilon ||
        std::abs(output[3] - 0.999329f) > epsilon ||
        std::abs(output[4] - (-0.999909f)) > epsilon) {
        return false;
        }

    //Tanh::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    Tanh::Derivative(&input2, &output2, nullptr, nullptr);

    if (std::abs(output2[0] - 0.419974f) > epsilon ||
        std::abs(output2[1] - 0.0706508f) > epsilon ||
        std::abs(output2[2] - 0.00986604f) > epsilon ||
        std::abs(output2[3] - 0.00134095f) > epsilon ||
        std::abs(output2[4] - 0.000181583f) > epsilon) {
        return false;
        }

    //Tanh::InitWeights()
    MAT<5,2,1>* weights = Tanh::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        delete weights;
        return false;
    }

    delete weights;
    return true;
}

//Test for
//  Softmax::FeedForward(Matrix*,Matrix*)
//  Softmax::Derivative(Matrix*,Matrix*)
//  Softmax::InitWeights()
bool ActivationTests::TestSoftmax()
{
    const float epsilon = 1e-6f;

    //Softmax::FeedForward(Matrix*,Matrix*)
    typedef Activation<Softmax<5,2,1,1>> Softmax;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    Softmax::FeedForward(&input, &output);

    if (std::abs(output[0] - 0.005894f) > epsilon ||
        std::abs(output[1] - 0.118392f) > epsilon ||
        std::abs(output[2] - 0.000798f) > epsilon ||
        std::abs(output[3] - 0.874808f) > epsilon ||
        std::abs(output[4] - 0.000108f) > epsilon) {
        return false;
        }

    //Softmax::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    Softmax::Derivative(&input2, &output2, nullptr, nullptr);

    if (std::abs(output2[0] - 1) > epsilon ||
        std::abs(output2[1] - 1) > epsilon ||
        std::abs(output2[2] - 1) > epsilon ||
        std::abs(output2[3] - 1) > epsilon ||
        std::abs(output2[4] - 1) > epsilon) {
        return false;
        }

    //Softmax::InitWeights()
    MAT<5,2,1>* weights = Softmax::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        delete weights;
        return false;
    }

    delete weights;
    return true;
}

//Test for
//  Sigmoid::FeedForward(Matrix*,Matrix*)
//  Sigmoid::Derivative(Matrix*,Matrix*)
//  Sigmoid::InitWeights()
bool ActivationTests::TestSigmoid()
{
    const float epsilon = 1e-6f;

    //Sigmoid::FeedForward(Matrix*,Matrix*)
    typedef Activation<Sigmoid<5,2,1,1>> Sigmoid;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    Sigmoid::FeedForward(&input, &output);


    if (std::abs(output[0] - 0.268941f) > epsilon ||
        std::abs(output[1] - 0.880797f) > epsilon ||
        std::abs(output[2] - 0.0474259f) > epsilon ||
        std::abs(output[3] - 0.982014f) > epsilon ||
        std::abs(output[4] - 0.00669285f) > epsilon) {
        return false;
        }
    //Sigmoid::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    Sigmoid::Derivative(&input2, &output2, nullptr, nullptr);

    if (std::abs(output2[0] - 0.196612f) > epsilon ||
        std::abs(output2[1] - 0.104994f) > epsilon ||
        std::abs(output2[2] - 0.0451767f) > epsilon ||
        std::abs(output2[3] - 0.0176627f) > epsilon ||
        std::abs(output2[4] - 0.00664806f) > epsilon) {
        return false;
        }
    //Sigmoid::InitWeights()
    MAT<5,2,1>* weights = Sigmoid::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        delete weights;
        return false;
    }

    delete weights;
    return true;
}

//Test for
//  SigmoidPrime::FeedForward(Matrix*,Matrix*)
//  SigmoidPrime::Derivative(Matrix*,Matrix*)
//  SigmoidPrime::InitWeights()
bool ActivationTests::TestSigmoidPrime()
{
    const float epsilon = 1e-6f;

    //SigmoidPrime::FeedForward(Matrix*,Matrix*)
    typedef Activation<SigmoidPrime<5,2,1,1>> SigmoidPrime;
    MAT<5,1,1> input({-1,2,-3,4,-5});
    MAT<5,1,1> output;
    SigmoidPrime::FeedForward(&input, &output);

    if (std::abs(output[0] - 0.196612f) > epsilon ||
        std::abs(output[1] - 0.104994f) > epsilon ||
        std::abs(output[2] - 0.0451767f) > epsilon ||
        std::abs(output[3] - 0.0176627f) > epsilon ||
        std::abs(output[4] - 0.00246651f) > epsilon) {
        return false;
        }

    //SigmoidPrime::Derivative(Matrix*,Matrix*)
    MAT<5,1,1> input2({-1,2,-3,4,-5});
    MAT<5,1,1> output2;
    SigmoidPrime::Derivative(&input2, &output2, nullptr, nullptr);

    if (std::abs(output2[0] - 0.196612f) > epsilon ||
        std::abs(output2[1] - 0.104994f) > epsilon ||
        std::abs(output2[2] - 0.0451767f) > epsilon ||
        std::abs(output2[3] - 0.0176627f) > epsilon ||
        std::abs(output2[4] - 0.00246651f) > epsilon) {
        return false;
        }

    //SigmoidPrime::InitWeights()
    MAT<5,2,1>* weights = SigmoidPrime::InitWeights();
    if(weights->GetRows() != 5 || weights->GetCols() != 2) {
        delete weights;
        return false;
    }

    delete weights;
    return true;
}





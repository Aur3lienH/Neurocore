#include "tests/LayerTests.h"
#include "network/layers/FCL.cuh"
#include "network/layers/InputLayer.cuh"
#include "network/layers/DropoutFCL.cuh"
#include "network/layers/MaxPooling.cuh"
#include "network/layers/AveragePooling.cuh"
#include "network/layers/ConvLayer.cuh"
#include "network/layers/Reshape.cuh"
#include <functional>
#include <iostream>
#include <ratio>


bool LayerTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));
    functions.emplace_back((void*)TestInputLayer, std::string("Input Layer"));
    functions.emplace_back((void*)TestFCLLayer, std::string("FCL layer"));
    functions.emplace_back((void*)TestCNNLayer, std::string("CNN layer"));


    functions.emplace_back((void*)TestDropLayer, std::string("Dropout Layer"));
    functions.emplace_back((void*)TestDropLayerBackprop, std::string("Dropout Layer Backprop"));
    functions.emplace_back((void*)TestMaxPoolLayer, std::string("Max Pooling Layer"));
    functions.emplace_back((void*)TestAveragePoolLayer, std::string("Average Pooling Layer"));
    functions.emplace_back((void*)TestMaxPoolLayerBackprop, std::string("Max Pooling Layer Backprop"));
    functions.emplace_back((void*)TestAveragePoolLayerBackprop, std::string("Average Pooling Layer Backprop"));

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
    ConvLayer<Activation<ReLU<1,3>>, LayerShape<3,3>, LayerShape<1,1>, LayerShape<3,3>, Constant<0.01>, true> cnn;
    cnn.Compile();

    Matrix<3,3> filters({0,0,0,0,1,0,0,0,0});
    cnn.SetWeights(&filters);

    Matrix<1,1> biases(0.);
    cnn.SetBiases(&biases);

    Matrix<3,3> input({1,1,1,1,2,1,1,1,1});

    Matrix<1,1>* out = cnn.FeedForward(&input);
    input.Print();
    filters.Print();
    out->Print();

    bool forwardPassCorrect = (std::abs((*out)(0,0) - 2.0) < 1e-6);

    if (!forwardPassCorrect)
    {
        std::cout << "we are here, over there" << std::endl;
        out->Print();
        return false;
    }

    Matrix<1,1> delta(1.0);


    Matrix<3,3>* bout = cnn.BackPropagate(&delta, &input);




    bool backwardPassCorrect = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float expected = (i == 1 && j == 1) ? 1.0 : 0.0;
            if (std::abs((*bout)(i,j) - expected) > 1e-6) {
                backwardPassCorrect = false;
                break;
            }
        }
    }

    cnn.getFilters()->Print();


    return backwardPassCorrect;
}


bool LayerTests::TestDropLayer()
{
    typedef LayerShape<3000,3000> DropLayerShape;
    Dropout<DropLayerShape,0.75> drop;
    LMAT<DropLayerShape> input(1);
    const LMAT<DropLayerShape> *out = drop.FeedForward(&input);
    size_t count = 0;
    for (int i = 0; i < out->GetRows(); i++)
    {
        for (int j = 0; j < out->GetCols(); j++)
        {
            if (out->data[j + i * out->GetCols()] < 1e-5)
            {
                count++;
            }
        }
    }
    float ratio = (float)count / (float)(out->GetRows() * out->GetCols());
    if (ratio < 0.70 || ratio > 0.80)
    {
        return false;
    }
    return true;
}

bool LayerTests::TestDropLayerBackprop()
{
    typedef LayerShape<3000,3000> DropLayerShape;
    Dropout<DropLayerShape,0.25> drop;

    LMAT<DropLayerShape> input(1);
    const LMAT<DropLayerShape> *forward_out = drop.FeedForward(&input);

    LMAT<DropLayerShape> gradient(1);
    LMAT<DropLayerShape> ones(1);
    const LMAT<DropLayerShape> *back_out = drop.BackPropagate(&gradient,&ones);

    size_t count = 0;
    for (int i = 0; i < back_out->GetRows(); i++)
    {
        for (int j = 0; j < back_out->GetCols(); j++)
        {
            if (back_out->data[j + i * back_out->GetCols()] < 1e-5)
            {
                count++;
            }
        }
    }

    float ratio = (float)count / (float)(back_out->GetRows() * back_out->GetCols());

    if (ratio < 0.20 || ratio > 0.30)
    {
        return false;
    }

    return true;
}


bool LayerTests::TestMaxPoolLayer()
{
    MaxPoolLayer<LayerShape<1,1>,LayerShape<3,3>,2,2> maxpool;
    maxpool.Compile();
    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    const Matrix<>* out = maxpool.FeedForward(&input);
    return out->data[0] - 5.0f < 1e-5;
}

bool LayerTests::TestAveragePoolLayer()
{
    AveragePoolLayer<LayerShape<1,1>,LayerShape<3,3>,2,2> avgpool;
    avgpool.Compile();
    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    const Matrix<>* out = avgpool.FeedForward(&input);
    return out->data[0] - 3.0f < 1e-5;
}

bool LayerTests::TestMaxPoolLayerBackprop()
{
    MaxPoolLayer<LayerShape<1,1>, LayerShape<3,3>, 2, 2> maxpool;
    maxpool.Compile();

    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    const Matrix<>* forward_out = maxpool.FeedForward(&input);

    Matrix<1,1> gradient({1.0f});

    // Backpropagate
    const Matrix<3,3>* back_out = maxpool.BackPropagate(&gradient, &input);

    bool isCorrect = true;
    float sum = 0.0f;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sum += back_out->data[j + i * back_out->GetCols()];

            if (input.data[j + i * input.GetCols()] != 5.0f &&
                back_out->data[j + i * back_out->GetCols()] > 1e-5) {
                isCorrect = false;
            }
        }
    }

    if (std::abs(sum - 1.0f) > 1e-5) {
        isCorrect = false;
    }

    return isCorrect;
}

bool LayerTests::TestAveragePoolLayerBackprop()
{
    AveragePoolLayer<LayerShape<1,1>, LayerShape<3,3>, 2, 2> avgpool;
    avgpool.Compile();

    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    const Matrix<>* forward_out = avgpool.FeedForward(&input);

    Matrix<1,1> gradient({1.0f});

    // Backpropagate
    const Matrix<3,3>* back_out = avgpool.BackPropagate(&gradient, &input);

    bool isCorrect = true;
    float expectedValue = 1.0f / (2.0f * 2.0f);
    float sum = 0.0f;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float value = back_out->data[j + i * back_out->GetCols()];
            sum += value;

            if (i < 2 && j < 2) {
                if (std::abs(value - expectedValue) > 1e-5) {
                    isCorrect = false;
                }
            } else {
                if (value > 1e-5) {
                    isCorrect = false;
                }
            }
        }
    }

    if (std::abs(sum - 1.0f) > 1e-5) {
        isCorrect = false;
    }

    return isCorrect;
}





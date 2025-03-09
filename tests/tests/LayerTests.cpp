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

#include "network/Network.h"
#include "network/loss/Loss.h"
#include "network/loss/MSE.cuh"


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
    functions.emplace_back((void*)TestCNNMultiple,std::string("CNN multiple layers, multiple filters dimensions"));

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
    ConvLayer<Activation<ReLU<1,3>>, LayerShape<3,3>, LayerShape<1,1>, LayerShape<3,3>, Constant<0.01>, GPU_DEFAULT ,true> cnn;
    cnn.Compile();

    Matrix<3,3> filters({0,0,0,0,1,0,0,0,0});
    cnn.SetWeights(&filters);

    Matrix<1,1> biases(0.);
    cnn.SetBiases(&biases);

    Matrix<3,3> input({1,1,1,1,2,1,1,1,1});

    Matrix<1,1>* out = cnn.FeedForward(&input);

    bool forwardPassCorrect = (std::abs((*out).get(0,0) - 2.0) < 1e-6);

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
            if (std::abs((*bout).get(i,j) - expected) > 1e-6) {
                backwardPassCorrect = false;
                break;
            }
        }
    }



    return backwardPassCorrect;
}

bool LayerTests::TestCNNLayerWeightsInit()
{
    typedef ConvLayer<Activation<ReLU<3,3,3,3>>, LayerShape<3,3,2>, LayerShape<3,3,3>, LayerShape<1,1,3>, Constant<0.01>, true> cnn2;
}

bool LayerTests::TestCNNMultiple()
{
    typedef ConvLayer<Activation<ReLU<3,3,3,2>>, LayerShape<3,3>, LayerShape<3,3,2>, LayerShape<1,1,2>, Constant<0.01>, GPU_DEFAULT,true> cnn1;
    typedef ConvLayer<Activation<ReLU<3,3,3,3>>, LayerShape<3,3,2>, LayerShape<3,3,3>, LayerShape<1,1,3>, Constant<0.01>, GPU_DEFAULT,true> cnn2;
    Network<MSE<3,3,3>,InputLayer<LayerShape<3,3,1>>,cnn1,cnn2> net;
    std::cout << "before compiling ! \n";
    net.Compile();
    std::cout << "after compiling \n";

    cnn1* conv1 = net.GetLayer<1>();
    cnn2* conv2 = net.GetLayer<2>();

    // For first layer: 1 input channel * 2 output channels = 2 filters
    // Each filter is 1x1
    MAT<1,1,2>* filters1 = new MAT<1,1,2>({0.5f, 1.0f});
    conv1->SetWeights(filters1);
    conv1->SetBiases(new MAT<1,1,2>({0.5f, 1.0f}));

    // For second layer: 2 input channels * 3 output channels = 6 filters
    // Each filter is 1x1
    MAT<1,1,6>* filters2 = new MAT<1,1,6>({
        2.0f, 0.5f,  // First input channel to output channels 1,2,3
        3.0f, 1.5f,  // Added values for the remaining connections
        0.8f, 1.2f   // Second input channel to output channels 1,2,3
    });
    conv2->SetWeights(filters2);
    conv2->SetBiases(new MAT<1,1,3>({1.0f, 2.0f, 3.0f}));

    MAT<3,3,1>* mat = new MAT<3,3,1>({1,0,1,0,1,0,1,0,1});
    std::cout << "after intializing the things ! \n";

    //Input is
    // 1 0 1
    // 0 1 0
    // 1 0 1

    const MAT<3,3,3>* outi = net.FeedForward(mat);
    auto* res = conv1->getResult();

    //The first convLayer with two output channels:
    // First channel (filter 0.5 + bias 0.5):
    // 1.0 0.5 1.0
    // 0.5 1.0 0.5
    // 1.0 0.5 1.0

    // Second channel (filter 1.0 + bias 1.0):
    // 2.0 1.0 2.0
    // 1.0 2.0 1.0
    // 2.0 1.0 2.0

    float correct_res_cnn1[] = {
        1.0f,0.5f,1.0f,0.5f,1.0f,0.5f,1.0f,0.5f,1.0f,
        2.0f,1.0f,2.0f,1.0f,2.0f,1.0f,2.0f,1.0f,2.0f
    };

    for (size_t i = 0; i < 18; i++) {
        if (std::abs(res->get(i) - correct_res_cnn1[i]) > 1e-6) {
            res->Print(0);
            res->Print(1);
            std::cout << "First layer output mismatch at index " << i
                      << ": expected " << correct_res_cnn1[i]
                      << ", got " << res->data[i] << std::endl;
            return false;
        }
    }

    //Same calculation here.

    auto* res2 = conv2->getResult();

    float correct_res_cnn2[] = {
        4.0f, 2.5f, 4.0f, 2.5f, 4.0f, 2.5f, 4.0f, 2.5f, 4.0f,
        8.0f, 5.0f, 8.0f, 5.0f, 8.0f, 5.0f, 8.0f, 5.0f, 8.0f,
        6.2f, 4.6f, 6.2f, 4.6f, 6.2f, 4.6f, 6.2f, 4.6f, 6.2f
    };


    for (size_t i = 0; i < 27; i++) {
        if (std::abs(res2->get(i) - correct_res_cnn2[i]) > 1e-6) {
            res2->Print();
            std::cout << "Second layer output mismatch at index " << i
                      << ": expected " << correct_res_cnn2[i]
                      << ", got " << res2->get(i) << std::endl;
            return false;
        }
    }

    Matrix<3,3,3>* expected = new Matrix<3,3,3>({
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.5f, 4.0f, 2.5f, 4.0f,
        8.0f, 5.0f, 8.0f, 5.0f, 8.0f, 0.0f, 0.0f, 5.0f, 8.0f,
        6.2f, 4.6f, 6.2f, 4.6f, 6.2f, 4.6f, 6.2f, 0.0f, 0.0f
    });

    net.BackPropagate(mat,expected);

    auto* delta = conv2->getDelta();
    auto* delta_bias = conv2->getDeltaBias();

    float* correct_delta_cnn2 = new float[]{14.5, 29.0, 10.5, 21.0, 8.5, 17.0};
    float* correct_delta_bias_cnn2 = new float[]{17, 13, 10.8};

    for (int i = 0; i < 6; i++) {
        if (std::abs((*delta).get(i) - correct_delta_cnn2[i]) > 1e-6) {
            std::cout << "the offset is : " << delta->GetOffset() << " \n";
            delta->PrintAllDims();
            std::cout << "the offset is : " << delta->GetOffset() << " \n";
            std::cout << "Second layer delta output mismatch at index " << i
                      << ": expected " << correct_delta_cnn2[i]
                      << ", got " << (*delta).get(i) << std::endl;
            return false;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (std::abs((*delta_bias).get(i) - correct_delta_bias_cnn2[i]) > 1e-6) {
            delta_bias->Print(i);
            std::cout << "Second layer delta bias output mismatch at index " << i
                      << ": expected " << correct_delta_bias_cnn2[i]
                      << ", got " << delta_bias->data[i] << std::endl;
            return false;
        }
    }


    return true;
}


bool LayerTests::TestDropLayer()
{
    if (GPU_DEFAULT)
        return false;
    typedef LayerShape<3000,3000> DropLayerShape;
    Dropout<DropLayerShape,0.75> drop;
    LMAT<DropLayerShape> input(1);
    const LMAT<DropLayerShape> *out = drop.FeedForward(&input);
    size_t count = 0;
    for (int i = 0; i < out->GetRows(); i++)
    {
        for (int j = 0; j < out->GetCols(); j++)
        {
            if (out->get(j + i * out->GetCols()) < 1e-5)
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
    if (GPU_DEFAULT)
        return false;
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
            if (back_out->get(j + i * back_out->GetCols()) < 1e-5)
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
    return out->get(0) - 5.0f < 1e-5;
}

bool LayerTests::TestAveragePoolLayer()
{
    AveragePoolLayer<LayerShape<1,1>,LayerShape<3,3>,2,2> avgpool;
    avgpool.Compile();
    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    const Matrix<>* out = avgpool.FeedForward(&input);
    return out->get(0) - 3.0f < 1e-5;
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
            sum += back_out->get(j + i * back_out->GetCols());

            if (input.get(j + i * input.GetCols()) != 5.0f &&
                back_out->get(j + i * back_out->GetCols()) > 1e-5) {
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
            float value = back_out->get(j + i * back_out->GetCols());
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





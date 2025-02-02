#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"

template<int rows,int prev_rows, int cols = 1, int dims = 1>
class Sigmoid final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;


    Sigmoid() {
#if USE_GPU
        checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
        checkCUDNN(
                cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
    }


    static double Function(double input) {
        return 1 / (1 + exp(-input));
    }

    static double Derive(double input)
    {
        return exp(-input) / pow(1 + exp(-input), 2);
    }

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        DefaultFeedForward<rows,cols,dims>(input, output, Function);
    }

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        DefaultDerivative<rows,cols,dims>(input, output, Derive);
    }

    static MAT<rows,prev_rows>* InitWeights()
    {
        auto* weights = new MAT<rows,prev_rows>();
        WeightsInit::XavierInit<rows,prev_rows,dims>(prev_rows, weights);
        return weights;
    }

    static std::string getName()
    {
        return "Sigmoid";
    }
};
#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"


template<int rows,int prev_rows, int cols = 1, int dims = 1>
class Tanh final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    Tanh();

#if not USE_GPU

    static double Function(double input);

#endif

    static double Derive(double input);

    static MAT<rows,prev_rows,dims>* InitWeights();


    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        DefaultFeedForward(input, output, Function);
    }

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        DefaultDerivative(input, output, Derive);
    }

    static std::string getName()
    {
        return "TanH";
    }
};

template<int rows,int prev_rows, int cols, int dims>
Tanh<rows,prev_rows,cols,dims>::Tanh()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU
template<int rows,int prev_rows, int cols, int dims>
double Tanh<rows,prev_rows,cols,dims>::Function(const double input)
{
    return tanh(input);
}

#endif
template<int rows,int prev_rows, int cols, int dims>
double Tanh<rows,prev_rows,cols,dims>::Derive(const double input)
{
    return 1 - tanh(input) * tanh(input);
}
template<int rows,int prev_rows, int cols, int dims>
MAT<rows,prev_rows,dims>* Tanh<rows,prev_rows,cols,dims>::InitWeights()
{
    auto* weights = new MAT<rows,prev_rows,dims>();

    WeightsInit::XavierInit(prev_rows, weights);
    return weights;
}
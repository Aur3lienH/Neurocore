#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include "Activation.cuh"


template<int rows,int prev_rows, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class Tanh final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    Tanh();

    static double Function(double input) requires(!GPU);

    static double Derive(double input);

    static MAT<rows,prev_rows,dims>* InitWeights();


    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        if constexpr(GPU)
        {
            cudnnActivationDescriptor_t activationDesc;
            checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
            checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0));
            DefaultFeedForward(input, output, &activationDesc);
            return;
        }
        else
            DefaultFeedForward(input, output, (void*)Function);
    }

    static void Derivative(const MAT<rows,cols,dims>* x_, MAT<rows,cols,dims>* dx_, const Matrix<rows,cols,dims>* dy_, const Matrix<rows,cols,dims>* y_)
    {
        if constexpr (GPU)
        {
            cudnnActivationDescriptor_t activationDesc; // Todo: move that elsewhere in a proper way
            checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
            checkCUDNN(
                cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0));
            DefaultDerivative(x_, dx_, &activationDesc, dy_, y_);
            return;
        }
        DefaultDerivative(x_, dx_, (void*)Derive, dy_, y_);
    }

    static std::string getName()
    {
        return "TanH";
    }
};

template<int rows,int prev_rows, int cols, int dims, bool GPU>
Tanh<rows,prev_rows,cols,dims, GPU>::Tanh()
{

}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
double Tanh<rows,prev_rows,cols,dims, GPU>::Function(const double input) requires(!GPU)
{
    return tanh(input);
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
double Tanh<rows,prev_rows,cols,dims, GPU>::Derive(const double input)
{
    return 1 - tanh(input) * tanh(input);
}
template<int rows,int prev_rows, int cols, int dims, bool GPU>
MAT<rows,prev_rows,dims>* Tanh<rows,prev_rows,cols,dims, GPU>::InitWeights()
{
    auto* weights = new MAT<rows,prev_rows,dims>();

    WeightsInit::XavierInit(prev_rows, weights);
    return weights;
}
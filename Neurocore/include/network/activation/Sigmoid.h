#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"

template<int rows,int prev_rows, int cols = 1, int dims = 1, bool GPU=GPU_DEFAULT>
class Sigmoid final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;


    static double Function(double input) requires(!GPU) {
        return 1 / (1 + exp(-input));
    }

    static double Derive(double input) requires(!GPU)
    {
        return exp(-input) / pow(1 + exp(-input), 2);
    }

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        if constexpr (GPU)
        {
            cudnnActivationDescriptor_t activationDesc;
            checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
            checkCUDNN(
                    cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
            DefaultFeedForward(input, output, &activationDesc);
            return;
        }
        else
            {DefaultFeedForward<rows,cols,dims>(input, output, (void*)Function);}
    }

    static void Derivative(const MAT<rows,cols,dims>* x_, MAT<rows,cols,dims>* dx_, const Matrix<rows,cols,dims>* dy_, const Matrix<rows,cols,dims>* y_)
    {
        if constexpr (GPU)
        {
            cudnnActivationDescriptor_t activationDesc; // Todo: move that elsewhere in a proper way
            checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
            checkCUDNN(
                cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
            DefaultDerivative(x_, dx_, &activationDesc, dy_, y_);
            return;
        }
        else
            {DefaultDerivative<rows,cols,dims>(x_, dx_, (void*)Derive, dy_, y_);}
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

private:
#if USE_GPU
    cudnnActivationDescriptor_t activationDesc;
#endif
};
#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include <type_traits>

template<int rows,int prev_rows, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class SigmoidPrime final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    SigmoidPrime();


    static double Function(double input) requires (!GPU);

    static double Derive(double input);

    static MAT<rows,prev_rows,dims>* InitWeights();

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
#if USE_GPU
        throw new std::runtime_error("Sigmoid prime not implemented on GPU");
        /*checkCUDNN(cudnnActivationForward(cuda->cudnnHandle, activationDesc, &cuda->one,
                                      input->desc, input->GetData(), &cuda->zero,
                                      output->desc, output->GetData()));*/
#else
        DefaultFeedForward(input, output, (void*)Function);
#endif
    }

    static void Derivative(const MAT<rows,cols,dims>* x_, MAT<rows,cols,dims>* dx_, const Matrix<rows,cols,dims>* dy_, const Matrix<rows,cols,dims>* y_)
    {
        DefaultDerivative(x_, dx_, (void*)Derive, dy_, y_);
    }

    static std::string getName()
    {
        return "SigmoidPrime";
    }

    static std::enable_if_t<dims == 1, cudnnActivationDescriptor_t> activationDesc;
};


template<int rows,int prev_rows, int cols, int dims, bool GPU>
SigmoidPrime<rows,prev_rows,cols,dims, GPU>::SigmoidPrime()
{
#if USE_GPU
    throw std::runtime_error("The sigmoid prime class has no meaning on GPU, please use the sigmoid class instead");
#endif
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
double SigmoidPrime<rows,prev_rows,cols,dims, GPU>::Function(double input) requires(!GPU)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

template<int rows,int prev_rows, int cols, int dims, bool GPU>
double SigmoidPrime<rows,prev_rows,cols,dims, GPU>::Derive(const double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}
template<int rows,int prev_rows, int cols, int dims, bool GPU>
MAT<rows,prev_rows,dims>* SigmoidPrime<rows,prev_rows,cols,dims, GPU>::InitWeights()
{
    auto* weights = new MAT<rows,prev_rows>();

    WeightsInit::XavierInit(prev_rows, weights);
    return weights;
}



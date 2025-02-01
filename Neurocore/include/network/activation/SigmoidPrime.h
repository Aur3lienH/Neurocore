#pragma once
#include <cmath>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"

template<int rows,int prev_rows, int cols = 1, int dims = 1>
class SigmoidPrime final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    SigmoidPrime();

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
        return "SigmoidPrime";
    }
};


template<int rows,int prev_rows, int cols, int dims>
SigmoidPrime<rows,prev_rows,cols,dims>::SigmoidPrime()
{
#if USE_GPU
    throw std::runtime_error("The sigmoid prime class has no meaning on GPU, please use the sigmoid class instead");
#endif
}

#if not USE_GPU
template<int rows,int prev_rows, int cols, int dims>
double SigmoidPrime<rows,prev_rows,cols,dims>::Function(double input)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

#endif
template<int rows,int prev_rows, int cols, int dims>
double SigmoidPrime<rows,prev_rows,cols,dims>::Derive(const double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}
template<int rows,int prev_rows, int cols, int dims>
MAT<rows,prev_rows,dims>* SigmoidPrime<rows,prev_rows,cols,dims>::InitWeights()
{
    auto* weights = new MAT<rows,prev_rows>();

    WeightsInit::XavierInit(prev_rows, weights);
    return weights;
}



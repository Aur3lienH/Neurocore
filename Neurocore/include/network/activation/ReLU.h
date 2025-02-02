#pragma once
#include <cmath>
#include <emmintrin.h>
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"

template<int rows,int prev_rows, int cols = 1, int dims = 1>
class ReLU final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;
public:
    ReLU();

#if not USE_GPU

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output);

    static double Function(double input);

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output);

#endif

    static double Derive(double input);

    static MAT<rows,prev_rows,dims>* InitWeights();

    static MAT<rows,cols,dims>* InitBiases();

    static std::string getName()
    {
        return "ReLU";
    }


};


template<int rows, int prev_rows, int cols, int dims>
ReLU<rows,prev_rows,cols,dims>::ReLU()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(
            cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

template<int rows, int prev_rows, int cols, int dims>
void ReLU<rows,prev_rows,cols,dims>::FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
{
    __m128 zero = _mm_setzero_ps();

    size_t i = 0;
    size_t s = input->GetSize();
    if (s > 4) // prevents size_t underflow
    {
        for (; i <= s - 4; i += 4)
        {
            __m128 vals = _mm_loadu_ps(&((*input)[i]));
            __m128 result = _mm_max_ps(zero, vals);
            _mm_storeu_ps(&((*output)[i]), result);
        }
    }

    // Process any remaining values
    for (; i < s; ++i)
    {
        if ((*input)[i] < 0) (*output)[i] = 0;
        else (*output)[i] = (*input)[i];
    }
}

#endif

#if not USE_GPU

template<int rows, int prev_rows, int cols, int dims>
void ReLU<rows,prev_rows,cols,dims>::Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
{
    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0);

    int i;
    for (i = 0; i <= input->GetSize() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 mask = _mm_cmpgt_ps(vals,
                                   zero); // Create a mask where each element is either 0xFFFFFFFFFFFFFFFF if vals > 0 or 0x0 otherwise
        __m128 result = _mm_and_ps(one, mask);  // Set to 1.0 where mask is true
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->GetSize(); ++i)
    {
        (*output)[i] = ((*input)[i] > 0) ? 1.0 : 0.0;
    }
}
template<int rows, int prev_rows, int cols, int dims>
double ReLU<rows,prev_rows,cols,dims>::Function(const double input)
{
    if (input > 0)
    {
        return input;
    }
    else
    {
        return 0;
    }
}

#endif
template<int rows, int prev_rows, int cols, int dims>
double ReLU<rows,prev_rows,cols,dims>::Derive(const double input)
{
    if (input > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

template<int rows, int prev_rows, int cols, int dims>
MAT<rows,prev_rows,dims>* ReLU<rows,prev_rows,cols,dims>::InitWeights()
{
    auto* weights = new MAT<rows,prev_rows,dims>();
    WeightsInit::HeUniform<rows,prev_rows,dims>(prev_rows, weights);
    return weights;
}

template<int rows, int prev_rows, int cols, int dims>
MAT<rows,cols,dims>* ReLU<rows,prev_rows,cols,dims>::InitBiases()
{
#if USE_GPU
    float* biases = new float[outputSize];
    for (int i = 0; i < outputSize; i++)
        biases[i] = 0.01f;

    Matrix_GPU* res = new Matrix_GPU(outputSize, 1);
    checkCUDA(cudaMemcpy(res->GetData(), biases, outputSize * sizeof(float), cudaMemcpyHostToDevice));
    delete[] biases;

    return res;
#else
    return new MAT<rows,cols,dims>(0.01f);
#endif
}
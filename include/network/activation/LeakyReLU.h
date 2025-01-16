#pragma once
#include <matrix/Matrix.cuh>
#include <network/InitFunc.cuh>
template<int rows,int prev_rows, float def_val = 0.01f, int cols = 1, int dims = 1>
class LeakyReLU final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    LeakyReLU();

#if not USE_GPU

    static double Function(double input);

#endif
    static double Derive(double input);

    static MAT<rows,prev_rows,dims>* InitWeights();

    //static void Save(std::ofstream& writer);

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
        return "LeakyReLU";
    }
};


template<int rows,int prev_rows, float def_val, int cols, int dims>
LeakyReLU<rows,prev_rows,def_val,cols,dims>::LeakyReLU()
{
#if USE_GPU
    throw std::runtime_error("LeakyReLU is not implemented on GPU");
#endif
}

#if not USE_GPU

template<int rows,int prev_rows, float def_val, int cols, int dims>
double LeakyReLU<rows,prev_rows,def_val,cols,dims>::Function(const double input)
{
    return input > 0 ? input : def_val * input;
}

#endif
template<int rows,int prev_rows, float def_val, int cols, int dims>
double LeakyReLU<rows,prev_rows,def_val,cols,dims>::Derive(const double input)
{
    return input > 0 ? 1 : def_val;
}
/*
void LeakyReLU::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<const char*>(&ActivationID<LeakyReLU>::value), sizeof(int));
    writer.write(reinterpret_cast<char*>(&alpha), sizeof(float));
}
*/
template<int rows,int prev_rows, float def_val, int cols, int dims>
MAT<rows,prev_rows,dims>* LeakyReLU<rows,prev_rows,def_val,cols,dims>::InitWeights()
{

    auto* weights = new Matrix<rows,prev_rows,dims>();
    WeightsInit::HeUniform(prev_rows, weights);
    return weights;
}
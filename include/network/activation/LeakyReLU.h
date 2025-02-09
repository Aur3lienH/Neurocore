#pragma once
#include <matrix/Matrix.cuh>
#include <network/InitFunc.cuh>
template<int rows,int prev_rows, float def_val = 0.01f, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class LeakyReLU final
{
public:

    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
    static constexpr int PrevRows = prev_rows;

    LeakyReLU();

    static double Function(double input) requires(!GPU);

    static double Derive(double input) requires(!GPU);

    static MAT<rows,prev_rows,dims>* InitWeights();

    //static void Save(std::ofstream& writer);

    static void FeedForward(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output)
    {
        if constexpr (GPU)
        {
            throw new std::runtime_error("Leaky Relu not implemented on GPU, a simple kernel suffice");
        }
		else
		    DefaultFeedForward(input, output, (void*)Function);
    }

    static void Derivative(const MAT<rows,cols,dims>* input, MAT<rows,cols,dims>* output, const Matrix<rows,cols,dims>* lastDelta, const Matrix<rows,cols,dims>* z)
    {
        if constexpr (GPU)
        {
            throw new std::runtime_error("Leaky Relu not implemented on GPU, a simple kernel suffice");
        }
        else
            {DefaultDerivative<rows,cols,dims>(input, output, (void*)Derive, lastDelta, z);}
    }

    static std::string getName()
    {
        return "LeakyReLU";
    }
};


template<int rows,int prev_rows, float def_val, int cols, int dims, bool GPU>
LeakyReLU<rows,prev_rows,def_val,cols,dims,GPU>::LeakyReLU()
{
    if constexpr (GPU)
        throw std::runtime_error("LeakyReLU is not implemented on GPU");
}

template<int rows,int prev_rows, float def_val, int cols, int dims, bool GPU>
double LeakyReLU<rows,prev_rows,def_val,cols,dims,GPU>::Function(const double input) requires(!GPU)
{
    return input > 0 ? input : def_val * input;
}

template<int rows,int prev_rows, float def_val, int cols, int dims, bool GPU>
double LeakyReLU<rows,prev_rows,def_val,cols,dims,GPU>::Derive(const double input) requires(!GPU)
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
template<int rows,int prev_rows, float def_val, int cols, int dims, bool GPU>
MAT<rows,prev_rows,dims>* LeakyReLU<rows,prev_rows,def_val,cols,dims,GPU>::InitWeights()
{

    auto* weights = new Matrix<rows,prev_rows,dims>();
    WeightsInit::HeUniform(prev_rows, weights);
    return weights;
}
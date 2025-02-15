#pragma once

__global__ void leakyReluFeedForward(float* input, float *output, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] < 0 ? alpha * input[idx] : input[idx];;
    }
}

__global__ void leakyReluDerivative(float *input, float* output, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] < 0 ? alpha : 1;;
    }
}

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
            {leakyReluFeedForward<<<CUDA_KERNEL_ARGS(cuda, input->GetSize())>>>(input->data_d, output->data_d,input->GetSize(), def_val);}
        else
		    {DefaultFeedForward(input, output, (void*)Function);}
    }

    static void Derivative(const MAT<rows,cols,dims>* x, MAT<rows,cols,dims>* dx_, const Matrix<rows,cols,dims>* dy_, const Matrix<rows,cols,dims>* y_)
    {
        if constexpr (GPU)
            {leakyReluDerivative<<<CUDA_KERNEL_ARGS(cuda, x->GetSize())>>>(x->data_d, dx_->data_d, x->GetSize(), def_val);}
        else
            {DefaultDerivative<rows,cols,dims>(x, dx_, (void*)Derive, dy_, y_);}
    }

    static std::string getName()
    {
        return "LeakyReLU";
    }
};


template<int rows,int prev_rows, float def_val, int cols, int dims, bool GPU>
LeakyReLU<rows,prev_rows,def_val,cols,dims,GPU>::LeakyReLU()
{

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
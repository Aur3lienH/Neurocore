#pragma once

#include "matrix/Matrix.cuh"
#include "network/activation/ReLU.h"
#include "network/activation/Sigmoid.h"
#include "network/activation/Softmax.h"
#include "network/activation/Tanh.h"
#include "network/activation/SigmoidPrime.h"
#include "network/activation/LeakyReLU.h"
#include "network/InitFunc.cuh"
#include <fstream>
#include <emmintrin.h>
#include <cmath>
#include <functional>

template<int rows, int prev_rows,int cols, int dims>
class Sigmoid;
template<int rows, int prev_rows,int cols, int dims>
class SigmoidPrime;
template<int rows, int prev_rows,int cols, int dims, bool GPU>
class ReLU;
template<int rows, int prev_rows, float def_val,int cols, int dims, bool GPU>
class LeakyReLU;
template<int rows, int prev_rows,int cols, int dims>
class SoftMax;
template<int rows, int prev_rows,int cols, int dims>
class Tanh;

template <typename... Args>
struct ActivationID {
    static constexpr uint value = 255; // Default ID
};

// Specializations for specific type combinations
template<int rows, int prev_rows,int cols, int dims>
struct ActivationID<Sigmoid<rows,prev_rows,cols,dims>> {
    static constexpr uint value = 0;
};

template<int rows, int prev_rows,int cols, int dims>
struct ActivationID<SigmoidPrime<rows,prev_rows,cols,dims>> {
    static constexpr uint value = 1;
};

template <int rows,int prev_rows ,int cols, int dims>
struct ActivationID<ReLU<rows,prev_rows,cols, dims>> {
    static constexpr uint value = 2;
};

template<int rows, int prev_rows, float def_val,int cols, int dims, bool GPU>
struct ActivationID<LeakyReLU<rows,prev_rows,def_val,cols,dims, GPU>> {
    static constexpr uint value = 3;
};

template<int rows, int prev_rows,int cols, int dims>
struct ActivationID<Tanh<rows,prev_rows,cols,dims>> {
    static constexpr uint value = 5;
};


//Each subclass of activation must have :
// -> void FeedForward(const MAT* input, MAT* output)
// To apply the values on a matrix
// -> void Derivative(const MAT* input, MAT* output)
// To apply the derivatives on a matrix
// -> MAT* InitWeights(int inputSize, int outputSize)


template<typename Derived,typename... Args>
class Activation final
{
    std::tuple<Args...> params;
    unsigned int id;
public:
    ~Activation() = default;

    template<int x=1, int y=1, int z=1>
    static void FeedForward(const MAT<x,y,z>* input, MAT<x,y,z>* output)
    {
        Derived::FeedForward(input, output);
    }

    static void Derivative(const MAT<Derived::Rows,Derived::Cols,Derived::Dims>* input, MAT<Derived::Rows,Derived::Cols,Derived::Dims>* output)
    {
        Derived::Derivative(input, output);
    }

    static MAT<Derived::Rows,Derived::PrevRows,Derived::Dims>* InitWeights()
    {
        return Derived::InitWeights();
    }

    static MAT<Derived::Rows,Derived::Cols,Derived::Dims>* InitBiases()
    {
        return new MAT<Derived::Rows,Derived::Cols,Derived::Dims>(0.01);
    }

    /*
    static Activation<Derived,Args...>* Read(std::ifstream& reader)
    {
        //int value = reader.read();
    }

    void Save(std::ofstream& write)
    {
        write.write(reinterpret_cast<char*>(&id), sizeof(int));
    }
    */



protected:
    Activation();
};

typedef double(*ActivationFunc)(double);
typedef double(*DerivativeFunc)(double);

template<int x=1, int y=1, int z=1, bool GPU=GPU_DEFAULT>
void DefaultFeedForward(const MAT<x,y,z>* input, MAT<x,y,z>* output, void *function)
{
#if SAFE
    if (input->GetCols() != output->GetCols() || input->GetRows() != output->GetRows() ||
        input->GetDims() != output->GetDims())
    {
        throw std::invalid_argument("activation::FeedForward : Both matrix must have the same shape !");
    }
#endif

    if constexpr (GPU)
    {
        cudnnActivationDescriptor_t* activationDesc = static_cast<cudnnActivationDescriptor_t*>(function);
        checkCUDNN(cudnnActivationForward(cuda->cudnnHandle, *activationDesc, &cuda->one,
                                         input->desc, input->GetData(), &cuda->zero,
                                         output->desc, output->GetData()));
    }
    else
    {
        for (int i = 0; i < input->GetSize(); i++)
        {
            ActivationFunc Func = reinterpret_cast<ActivationFunc>(function);
            output[0][i] = Func(input[0][i]);
        }
    }
}

template<int x=1, int y=1, int z=1, bool GPU=GPU_DEFAULT>
void DefaultDerivative(const MAT<x,y,z>* input, MAT<x,y,z>* output, void* derivative, const Matrix<x,y,z>* lastDelta, const Matrix<x,y,z>* z_)
{
#if SAFE
    if (input->GetCols() != output->GetCols() || input->GetRows() != output->GetRows() ||
        input->GetDims() != output->GetDims())
    {
        throw std::invalid_argument("activation::Derivative() : Both matrix must have the same shape !");
    }
#endif

    if constexpr (GPU)
    {
        cudnnActivationDescriptor_t* activationDesc = static_cast<cudnnActivationDescriptor_t*>(derivative);
        checkCUDNN(cudnnActivationBackward(cuda->cudnnHandle, *activationDesc, &cuda->one,
                                       input->desc, input->GetData(),
                                       lastDelta->desc,
                                       lastDelta->GetData(), z_->desc, z_->GetData(),
                                       &cuda->zero, output->desc, output->GetData()));
    }
    else
    {
        for (int i = 0; i < input->GetSize(); i++)
        {
            DerivativeFunc Derive = reinterpret_cast<DerivativeFunc>(derivative);
            output[0][i] = Derive(input[0][i]);
        }
    }
}







/*
class None : public activation
{
public:
    None();

#if not USE_GPU

    double Function(double input) override;

#endif

    double Derivative(double input);

    MAT* InitWeigths(int inputSize, int outputSize);
};
*/

template<typename Derived,typename ... Args>
    Activation<Derived,Args...>::Activation() : id(ActivationID<Args...>::value)
{
}

/*
template<typename Derived,typename ... Args>
Activation<Args>* Activation<Derived,Args...>::Read(std::ifstream& reader)
{
    int ID;
    reader.read(reinterpret_cast<char*>(&ID), sizeof(int));
    if (ID == 0)
    {
        return new Activation<Sigmoid>();
    }
    else if (ID == 1)
    {
        return new Activation<SigmoidPrime>();
    }
    else if (ID == 2)
    {
        return new Activation<ReLU>();
    }
    else if (ID == 3)
    {
        float f;
        reader.read(reinterpret_cast<char*>(&f), sizeof(float));
        return new Activation<LeakyReLU,f>();
    }
    else if (ID == 4)
    {
        return new Softmax();
    }
    else if (ID == 5)
    {
        return new Tanh();
    }
    else
    {
        throw std::invalid_argument("Invalid ID for loading activation function");
    }
}
*/







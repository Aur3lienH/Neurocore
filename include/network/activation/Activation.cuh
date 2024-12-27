#pragma once

#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include <fstream>
#include <emmintrin.h>
#include <cmath>

class Sigmoid;
class SigmoidPrime;
class ReLU;
class LeakyReLU;
class SoftMax;
class Tanh;

template <typename... Args>
struct ActivationID {
    static constexpr uint value = 255; // Default ID
};

// Specializations for specific type combinations
template <>
struct ActivationID<Sigmoid> {
    static constexpr uint value = 0;
};

template <>
struct ActivationID<SigmoidPrime> {
    static constexpr uint value = 1;
};

template <>
struct ActivationID<ReLU> {
    static constexpr uint value = 2;
};

template <>
struct ActivationID<LeakyReLU> {
    static constexpr uint value = 3;
};

template <>
struct ActivationID<Tanh> {
    static constexpr uint value = 5;
};


//Each subclass of activation must have :
// -> void FeedForward(const MAT* input, MAT* output)
// To apply the values on a matrix
// -> void Derivative(const MAT* input, MAT* output)
// To apply the derivatives on a matrix
// -> MAT* InitWeights(int inputSize, int outputSize)


template<typename Derived,typename... Args>
class Activation
{
    std::tuple<Args...> params;
    unsigned int id;
public:
    virtual ~Activation() = default;

#if USE_GPU

    virtual void FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                             const cudnnTensorDescriptor_t& outputDesc);

    virtual void Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                            const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z,
                            const cudnnTensorDescriptor_t& zDesc,
                            MAT* output, const cudnnTensorDescriptor_t& outputDesc);

#else

    void FeedForward(const MAT* input, MAT* output)
    {
        static_cast<Derived*>(this)->FeedForward(input, output);
    }

    void Derivative(const MAT* input, MAT* output)
    {
        static_cast<Derived*>(this)->Derivative(input, output);
    }

#endif

    MAT* InitWeights(int inputSize, int outputSize)
    {
        return static_cast<Derived*>(this)->InitWeights(inputSize, outputSize);
    }

    MAT* InitBiases(int outputSize)
    {
        return new MAT(outputSize,1);
    }

    static Activation<Derived,Args...>* Read(std::ifstream& reader)
    {
        //int value = reader.read();
    }

    void Save(std::ofstream& write)
    {
        write.write(reinterpret_cast<char*>(&id), sizeof(int));
    }



protected:
    Activation();

#if USE_GPU

    void Function(const MAT& input, const cudnnTensorDescriptor_t& inputDesc, MAT& output,
                  const cudnnTensorDescriptor_t& outputDesc);

#else

#endif

#if USE_GPU
    cudnnActivationDescriptor_t activationDesc;
#endif
};


#if USE_GPU
template<typename ... Args>
voidActivation<Args...>::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                             const cudnnTensorDescriptor_t& outputDesc)
#else

void DefaultFeedForward(const MAT* input, MAT* output, double (*Function)(double))
#endif
{
#if SAFE
    if (input->GetCols() != output->GetCols() || input->GetRows() != output->GetRows() ||
        input->GetDims() != output->GetDims())
    {
        throw std::invalid_argument("activation::FeedForward : Both matrix must have the same shape !");
    }
#endif

#if USE_GPU
    checkCUDNN(cudnnActivationForward(Matrix_GPU::cuda->cudnnHandle, activationDesc, &Matrix_GPU::cuda->one,
                                      inputDesc, input->GetData(), &Matrix_GPU::cuda->zero,
                                      outputDesc, output->GetData()));
#else
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = Function(input[0][i]);
    }
#endif
}


#if USE_GPU

voidActivation<Args...>::Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                            const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z,
                            const cudnnTensorDescriptor_t& zDesc,
                            MAT* output, const cudnnTensorDescriptor_t& outputDesc)
#else

void DefaultDerivative(const MAT* input, MAT* output, double (*Derive)(double))
#endif
{
#if USE_GPU
    checkCUDNN(cudnnActivationBackward(Matrix_GPU::cuda->cudnnHandle, activationDesc, &Matrix_GPU::cuda->one,
                                       inputDesc, input->GetData(),
                                       lastDeltaDesc,
                                       lastDelta->GetData(), zDesc, z->GetData(),
                                       &Matrix_GPU::cuda->zero, outputDesc, output->GetData()));

#else

    if (input->GetCols() != output->GetCols() || input->GetRows() != output->GetRows() ||
        input->GetDims() != output->GetDims())
    {
        throw std::invalid_argument("activation::Derivative() : Both matrix must have the same shape !");
    }

    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = Derive(input[0][i]);
    }
#endif
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


#if USE_GPU

void Activation<Args...>::Function(const MAT& input, const cudnnTensorDescriptor_t& inputDesc, MAT& output,
                          const cudnnTensorDescriptor_t& outputDesc)
{
    checkCUDNN(cudnnActivationForward(Matrix_GPU::cuda->cudnnHandle, activationDesc, &Matrix_GPU::cuda->one,
                                      inputDesc, input.GetData(), &Matrix_GPU::cuda->zero,
                                      outputDesc, output.GetData()));
}

#endif







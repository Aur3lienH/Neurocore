#pragma once

#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include <fstream>
#include <emmintrin.h>
#include <cmath>

template <typename... Args>
struct ActivationID {
    static constexpr uint value = 0; // Default ID
};

// Specializations for specific type combinations
template <>
struct ActivationID<int> {
    static constexpr uint value = 1;
};

template <>
struct ActivationID<float, int> {
    static constexpr uint value = 2;
};

template <>
struct ActivationID<double, double> {
    static constexpr uint value = 3;
};
template<typename... Args>
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

    virtual void FeedForward(const MAT* input, MAT* output);

    virtual void Derivative(const MAT* input, MAT* output);

#endif

    virtual MAT* InitWeights(int inputSize, int outputSize) = 0;

    virtual MAT* InitBiases(int outputSize);

    static Activation<Args>* Read(std::ifstream& reader);

    virtual void Save(std::ofstream& write);

protected:
    Activation();

#if USE_GPU

    void Function(const MAT& input, const cudnnTensorDescriptor_t& inputDesc, MAT& output,
                  const cudnnTensorDescriptor_t& outputDesc);

#else

    virtual double Function(double input) = 0;

#endif

    virtual double Derive(double input) = 0;

#if USE_GPU
    cudnnActivationDescriptor_t activationDesc;
#endif
};

class Sigmoid
{
public:
    Sigmoid();

#if not USE_GPU

    double Function(double input);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);
};

class SigmoidPrime
{
public:
    SigmoidPrime();

#if not USE_GPU

    double Function(double input);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);
};

class ReLU
{
public:
    ReLU();

#if not USE_GPU

    void Derivative(const MAT* input, MAT* output);

    double Function(double input);

    void FeedForward(const MAT* input, MAT* output);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);

    MAT* InitBiases(int outputSize);
};

class LeakyReLU
{
public:
    explicit LeakyReLU(double alpha);

#if not USE_GPU

    double Function(double input);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);

    void Save(std::ofstream& writer);

private:
    double alpha;
};


class Softmax
{
public:
    Softmax();

#if USE_GPU

    void FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                     const cudnnTensorDescriptor_t& outputDesc) override;

    void Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                    const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z, const cudnnTensorDescriptor_t& zDesc,
                    MAT* output, const cudnnTensorDescriptor_t& outputDesc) override;

#else

    void FeedForward(const MAT* input, MAT* output);

    void Derivative(const MAT* input, MAT* output);

    double inline Function(double input)
    { return 0; };

#endif

    double inline Derive(double input)
    { return 0; };

    MAT* InitWeights(int inputSize, int outputSize);

};

class Tanh
{
public:
    Tanh();

#if not USE_GPU

    double Function(double input);

#endif

    double Derive(double input);

    MAT* InitWeights(int inputSize, int outputSize);
};


/*
class None : public Activation
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

template<typename ... Args>
Activation<Args...>::Activation() : id(ActivationID<Args...>::value)
{
}

template<typename ... Args>
MAT* Activation<Args...>::InitBiases(const int outputSize)
{
    return new MAT(outputSize, 1, 1);
}

#if USE_GPU
template<typename ... Args>
voidActivation<Args...>::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                             const cudnnTensorDescriptor_t& outputDesc)
#else

template<typename ... Args>
void Activation<Args...>::FeedForward(const MAT* input, MAT* output)
#endif
{
#if SAFE
    if (input->GetCols() != output->GetCols() || input->GetRows() != output->GetRows() ||
        input->GetDims() != output->GetDims())
    {
        throw std::invalid_argument("Activation::FeedForward : Both matrix must have the same shape !");
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

template<typename ... Args>
void Activation<Args...>::Derivative(const MAT* input, MAT* output)
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
        throw std::invalid_argument("Activation::Derivative() : Both matrix must have the same shape !");
    }

    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = Derive(input[0][i]);
    }
#endif
}

template<typename ... Args>
void Activation<Args...>::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&id), sizeof(int));
}

template<typename ... Args>
Activation<Args>* Activation<Args...>::Read(std::ifstream& reader)
{
    int ID;
    reader.read(reinterpret_cast<char*>(&ID), sizeof(int));
    if (ID == 0)
    {
        return new Sigmoid();
    }
    else if (ID == 1)
    {
        return new SigmoidPrime();
    }
    else if (ID == 2)
    {
        return new ReLU();
    }
    else if (ID == 3)
    {
        float f;
        reader.read(reinterpret_cast<char*>(&f), sizeof(float));
        return new LeakyReLU(f);
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

#if USE_GPU

void Activation<Args...>::Function(const MAT& input, const cudnnTensorDescriptor_t& inputDesc, MAT& output,
                          const cudnnTensorDescriptor_t& outputDesc)
{
    checkCUDNN(cudnnActivationForward(Matrix_GPU::cuda->cudnnHandle, activationDesc, &Matrix_GPU::cuda->one,
                                      inputDesc, input.GetData(), &Matrix_GPU::cuda->zero,
                                      outputDesc, output.GetData()));
}

#endif

Sigmoid::Sigmoid()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(
            cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

double Sigmoid::Function(const double input)
{
    return 1 / (1 + exp(-input));
}

#endif

double Sigmoid::Derive(const double input)
{
    return exp(-input) / pow(1 + exp(-input), 2);
}

MAT* Sigmoid::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new MAT(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}

SigmoidPrime::SigmoidPrime()
{
#if USE_GPU
    throw std::runtime_error("The sigmoid prime class has no meaning on GPU, please use the sigmoid class instead");
#endif
}

#if not USE_GPU

double SigmoidPrime::Function(double input)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

#endif

double SigmoidPrime::Derive(const double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}

MAT* SigmoidPrime::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}

ReLU::ReLU()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(
            cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

void ReLU::FeedForward(const MAT* input, MAT* output)
{
    __m128 zero = _mm_setzero_ps();

    size_t i;
    for (i = 0; i <= input->GetSize() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 result = _mm_max_ps(zero, vals);
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->GetSize(); ++i)
    {
        if ((*input)[i] < 0) (*output)[i] = 0;
    }
}

#endif

#if not USE_GPU

void ReLU::Derivative(const MAT* input, MAT* output)
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

double ReLU::Function(const double input)
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

double ReLU::Derive(const double input)
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

MAT* ReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::HeUniform(previousNeuronsCount, weights);
    return weights;
}

MAT* ReLU::InitBiases(const int outputSize)
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
    return new MAT(outputSize, 1, 0.01f);
#endif
}

LeakyReLU::LeakyReLU(const double _alpha)
{
    alpha = _alpha;
#if USE_GPU
    throw std::runtime_error("LeakyReLU is not implemented on GPU");
#endif
}

#if not USE_GPU

double LeakyReLU::Function(const double input)
{
    return input > 0 ? input : 0.01 * input;
}

#endif

double LeakyReLU::Derive(const double input)
{
    return input > 0 ? 1 : 0.01;
}

void LeakyReLU::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<const char*>(&ActivationID<LeakyReLU>::value), sizeof(int));
    writer.write(reinterpret_cast<char*>(&alpha), sizeof(float));
}

MAT* LeakyReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::HeUniform(previousNeuronsCount, weights);
    return weights;
}

Softmax::Softmax()
{
}

#if USE_GPU

void Softmax::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                          const cudnnTensorDescriptor_t& outputDesc)
#else

void Softmax::FeedForward(const MAT* input, MAT* output)
#endif
{
#if USE_GPU
    checkCUDNN(cudnnSoftmaxForward(Matrix_GPU::cuda->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &Matrix_GPU::cuda->one, inputDesc, input->GetData(),
                                   &Matrix_GPU::cuda->zero, outputDesc, output->GetData()));
#else
    double sum = 0;
    double max = input[0][0];
    for (int i = 0; i < input->GetSize(); i++)
    {
        if (input[0][i] > max)
        {
            max = input[0][i];
        }
    }

    for (int i = 0; i < input->GetSize(); i++)
    {
        sum += exp(input[0][i] - max);
    }
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = exp(input[0][i] - max) / sum;
    }
#endif
}

MAT* Softmax::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    MAT* weights = new MAT(NeuronsCount, previousNeuronsCount);
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}

#if USE_GPU

void Softmax::Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                         const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z,
                         const cudnnTensorDescriptor_t& zDesc,
                         MAT* output, const cudnnTensorDescriptor_t& outputDesc)
{
    /*checkCUDNN(cudnnSoftmaxBackward(Matrix_GPU::cuda->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &Matrix_GPU::cuda->one, *input->GetDescriptor_1D(), input->GetData(),
                                    *lastDelta->GetDescriptor_1D(), lastDelta->GetData(), &Matrix_GPU::cuda->zero,
                                    *output->GetDescriptor_1D(), output->GetData()));*/

    // The CPU version sets all values of output to one, but as the GPU version of Derivative also multiplies output
    // by lastDelta, we can just copy lastDelta to output
    checkCUDA(cudaMemcpy(output->GetData(), lastDelta->GetData(), output->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
}

#else

void Softmax::Derivative(const MAT* input, MAT* output)
{
    for (int i = 0; i < input->GetSize(); i++)
    {
        output[0][i] = 1;
    }
}

#endif


Tanh::Tanh()
{
#if USE_GPU
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0));
#endif
}

#if not USE_GPU

double Tanh::Function(const double input)
{
    return tanh(input);
}

#endif

double Tanh::Derive(const double input)
{
    return 1 - tanh(input) * tanh(input);
}

MAT* Tanh::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
#if USE_GPU
    auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
#else
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
#endif
    WeightsInit::XavierInit(previousNeuronsCount, weights);
    return weights;
}

/*
None::None() : Activation()
{

}

#if not USE_GPU

double None::Function(const double input)
{
    return 0;
}

#endif
*/








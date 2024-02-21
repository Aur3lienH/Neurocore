#include "network/Activation.cuh"
#include "network/InitFunc.cuh"
#include <fstream>
#include <emmintrin.h>
#include <cmath>


Activation::Activation()
{
    ID = -1;
}

std::string Activation::getName() const
{
    return name;
}

MAT* Activation::InitBiases(const int outputSize)
{
    return new MAT(outputSize, 1, 1);
}

#if USE_GPU

void Activation::FeedForward(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, MAT* output,
                             const cudnnTensorDescriptor_t& outputDesc)
#else

void Activation::FeedForward(const MAT* input, MAT* output)
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

void Activation::Derivative(const MAT* input, const cudnnTensorDescriptor_t& inputDesc, const MAT* lastDelta,
                            const cudnnTensorDescriptor_t& lastDeltaDesc, const MAT* z,
                            const cudnnTensorDescriptor_t& zDesc,
                            MAT* output, const cudnnTensorDescriptor_t& outputDesc)
#else

void Activation::Derivative(const MAT* input, MAT* output)
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

void Activation::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&ID), sizeof(int));
}

Activation* Activation::Read(std::ifstream& reader)
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

void Activation::Function(const MAT& input, const cudnnTensorDescriptor_t& inputDesc, MAT& output,
                          const cudnnTensorDescriptor_t& outputDesc)
{
    checkCUDNN(cudnnActivationForward(Matrix_GPU::cuda->cudnnHandle, activationDesc, &Matrix_GPU::cuda->one,
                                      inputDesc, input.GetData(), &Matrix_GPU::cuda->zero,
                                      outputDesc, output.GetData()));
}

#endif

Sigmoid::Sigmoid()
{
    name = "Sigmoid";
    ID = 0;
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
    name = "SigmoidPrime";
    ID = 1;
#if USE_GPU
    throw std::runtime_error("The sigmoid prime class has no meaning on GPU, please use the sigmoid class instead");
#endif
}

#if not USE_GPU

double SigmoidPrime::Function(const double input)
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
    name = "ReLU";
    ID = 2;
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
    ID = 3;
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
    writer.write(reinterpret_cast<char*>(&ID), sizeof(int));
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
    name = "Softmax";
    ID = 4;
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
    name = "Tanh";
    ID = 5;
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








#include "Activation.cuh"
#include "InitFunc.cuh"
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

Matrix* Activation::InitBiases(const int outputSize)
{
    auto* Biases = new Matrix(outputSize, 1);
    for (int i = 0; i < outputSize; i++)
    {
        Biases[0][i] = 0;
    }

    return Biases;
}

void Activation::FeedForward(const Matrix* input, Matrix* output)
{
    if (input->getCols() != output->getCols() || input->getRows() != output->getRows() ||
        input->getDim() != output->getDim())
    {
        throw std::invalid_argument("Activation::FeedForward : Both matrix must have the same shape !");
    }

    for (int i = 0; i < input->size(); i++)
    {
        output[0][i] = Function(input[0][i]);
    }
}

void Activation::Derivative(const Matrix* input, Matrix* output)
{
    if (input->getCols() != output->getCols() || input->getRows() != output->getRows() ||
        input->getDim() != output->getDim())
    {
        throw std::invalid_argument("Activation::Derivative() : Both matrix must have the same shape !");
    }

    for (int i = 0; i < input->size(); i++)
    {
        output[0][i] = Derive(input[0][i]);
    }
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

Sigmoid::Sigmoid()
{
    name = "Sigmoid";
    ID = 0;
}

double Sigmoid::Function(const double input)
{
    return 1 / (1 + exp(-input));
}

double Sigmoid::Derive(const double input)
{
    return exp(-input) / pow(1 + exp(-input), 2);
}

Matrix* Sigmoid::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount,1,true);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

SigmoidPrime::SigmoidPrime()
{
    name = "SigmoidPrime";
    ID = 1;
}

double SigmoidPrime::Function(const double input)
{
    return 0.5 + 0.5 * tanh(0.5 * input);
}

double SigmoidPrime::Derive(const double input)
{
    return 0.5 * (1 + tanh(0.5 * input)) * (1 - tanh(0.5 * input));
}

Matrix* SigmoidPrime::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

ReLU::ReLU()
{
    name = "ReLU";
    ID = 2;
}


void ReLU::FeedForward(const Matrix* input, Matrix* output)
{
    __m128 zero = _mm_setzero_ps();

    size_t i;
    for (i = 0; i <= input->size() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 result = _mm_max_ps(zero, vals);
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->size(); ++i)
    {
        if ((*input)[i] < 0) (*output)[i] = 0;
    }
}

void ReLU::Derivative(const Matrix* input, Matrix* output)
{
    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0);

    size_t i;
    for (i = 0; i <= input->size() - 4; i += 4)
    {
        __m128 vals = _mm_loadu_ps(&((*input)[i]));
        __m128 mask = _mm_cmpgt_ps(vals, zero); // Create a mask where each element is either 0xFFFFFFFFFFFFFFFF if vals > 0 or 0x0 otherwise
        __m128 result = _mm_and_ps(one, mask);  // Set to 1.0 where mask is true
        _mm_storeu_ps(&((*output)[i]), result);
    }

    // Process any remaining values
    for (; i < input->size(); ++i)
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

Matrix* ReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount,1,true);
    HeInit(previousNeuronsCount, weights);
    return weights;
}

LeakyReLU::LeakyReLU(const double _alpha)
{
    alpha = _alpha;
    ID = 3;
}

double LeakyReLU::Function(const double input)
{
    return input > 0 ? input : 0.01 * input;
}

double LeakyReLU::Derive(const double input)
{
    return input > 0 ? 1 : 0.01;
}

void LeakyReLU::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&ID), sizeof(int));
    writer.write(reinterpret_cast<char*>(&alpha), sizeof(float));
}

Matrix* LeakyReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount,1,true);
    HeInit(previousNeuronsCount, weights);
    return weights;
}

Softmax::Softmax()
{
    name = "Softmax";
    ID = 4;
}

void Softmax::FeedForward(const Matrix* input, Matrix* output)
{
    double sum = 0;
    double max = input[0][0];
    for (int i = 0; i < input->size(); i++)
    {
        if (input[0][i] > max)
        {
            max = input[0][i];
        }
    }

    for (int i = 0; i < input->size(); i++)
    {
        sum += exp(input[0][i] - max);
    }
    for (int i = 0; i < input->size(); i++)
    {
        output[0][i] = exp(input[0][i] - max) / sum;
    }
}

void Softmax::Derivative(const Matrix* input, Matrix* output)
{
    for (int i = 0; i < input->size(); i++)
    {
        output[0][i] = 1;
    }
}

Matrix* Softmax::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount,1,true);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}

double Softmax::Function(const double input)
{
    return 0;
}

double Softmax::Derive(const double input)
{
    return 0;
}

Tanh::Tanh()
{
    name = "Tanh";
    ID = 5;
}

double Tanh::Function(const double input)
{
    return tanh(input);
}

double Tanh::Derive(const double input)
{
    return 1 - tanh(input) * tanh(input);
}

Matrix* Tanh::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
{
    auto* weights = new Matrix(NeuronsCount, previousNeuronsCount,1,true);
    XavierInit(previousNeuronsCount, weights);
    return weights;
}


None::None() : Activation()
{

}

double None::Function(const double input)
{
    return 0;
}









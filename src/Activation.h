#pragma once

#include "Matrix.h"


//Abstact class to make all activations function
class Activation
{
public:

    virtual void FeedForward(const Matrix* input, Matrix* output);

    virtual void Derivative(const Matrix* input, Matrix* output);

    virtual Matrix* InitWeights(int inputSize, int outputSize) = 0;

    static Activation* Read(std::ifstream& reader);

    virtual void Save(std::ofstream& write);

    static Matrix* InitBiases(int outputSize);

    [[nodiscard]] std::string getName() const;

protected:
    Activation();

    virtual double Function(double input) = 0;

    virtual double Derive(double input) = 0;

    std::string name;
    int ID;

};


class Sigmoid : public Activation
{
public:
    Sigmoid();

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;
};

class SigmoidPrime : public Activation
{
public:
    SigmoidPrime();

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;
};

class ReLU : public Activation
{
public:
    ReLU();

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;
};

class LeakyReLU : public Activation
{
public:
    explicit LeakyReLU(double alpha);

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;

    void Save(std::ofstream& writer) override;

private:
    double alpha;
};


class Softmax : public Activation
{
public:
    Softmax();

    void FeedForward(const Matrix* input, Matrix* output) override;

    void Derivative(const Matrix* input, Matrix* output) override;

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;

};

class Tanh : public Activation
{
public:
    Tanh();

    double Function(double input) override;

    double Derive(double input) override;

    Matrix* InitWeights(int inputSize, int outputSize) override;
};

class None : public Activation
{
public:
    None();

    double Function(double input) override;

    double Derivative(double input);

    Matrix* InitWeigths(int inputSize, int outputSize);
};

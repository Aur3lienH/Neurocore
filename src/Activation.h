#pragma once
#include "Matrix.h"



//Abstact class to make all activations function
class Activation
{
public:
    Activation();
    virtual void FeedForward(const Matrix* input, Matrix* output);
    virtual void Derivative(const Matrix * input, Matrix* output);
    virtual Matrix* InitWeights(int inputSize, int outputSize) = 0;
    static Activation* Read(std::ifstream& reader);
    virtual void Save(std::ofstream& write);
    Matrix* InitBiases(int outputSize);
    std::string getName() const;
protected: 
    virtual double Function(double input) = 0;
    virtual double Derivate(double input) = 0;
    std::string name;
    int ID;
    
};


class Sigmoid : public Activation
{
public:
    Sigmoid();
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);
};

class SigmoidPrime : public Activation
{
public:
    SigmoidPrime();
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);
};

class ReLU : public Activation
{
public:
    ReLU();
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);
};

class LeakyReLU : public Activation
{
public:
    LeakyReLU(double alpha);
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);
    void Save(std::ofstream& writer);
private:
    double alpha;
};


class Softmax : public Activation
{
public:
    Softmax(); 
    void FeedForward(const Matrix* input, Matrix* output);
    void Derivative(const Matrix * input, Matrix* output);
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);

};

class Tanh : public Activation
{
public:
    Tanh();
    double Function(double input);
    double Derivate(double input);
    Matrix* InitWeights(int inputSize, int outputSize);
};

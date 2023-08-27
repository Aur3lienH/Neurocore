#pragma once

#include "Matrix.cuh"


//Abstact class to make all activations function
class Activation
{
public:
    virtual ~Activation() = default;

    virtual void FeedForward(const MAT* input, MAT* output);

#if USE_GPU

    virtual void Derivative(const MAT* input, const MAT* lastDelta, const MAT* z, MAT* output);

#else

    virtual void Derivative(const MAT* input, MAT* output);

#endif

    virtual MAT* InitWeights(int inputSize, int outputSize) = 0;

    static MAT* InitBiases(int outputSize);

    static Activation* Read(std::ifstream& reader);

    virtual void Save(std::ofstream& write);

    [[nodiscard]] std::string getName() const;

protected:
    Activation();

#if USE_GPU

    void Function(const MAT& input, MAT& output);

#else

    virtual double Function(double input) = 0;

#endif

    virtual double Derive(double input) = 0;

    std::string name;
    int ID;

#if USE_GPU
    cudnnActivationDescriptor_t activationDesc;
#endif
};


class Sigmoid : public Activation
{
public:
    Sigmoid();

#if not USE_GPU

    double Function(double input) override;

#endif

    double Derive(double input) override;

    MAT* InitWeights(int inputSize, int outputSize) override;
};

class SigmoidPrime : public Activation
{
public:
    SigmoidPrime();

#if not USE_GPU

    double Function(double input) override;

#endif

    double Derive(double input) override;

    MAT* InitWeights(int inputSize, int outputSize) override;
};

class ReLU : public Activation
{
public:
    ReLU();

#if not USE_GPU

    void Derivative(const MAT* input, MAT* output) override;

    double Function(double input) override;

    void FeedForward(const MAT* input, MAT* output) override;

#endif

    double Derive(double input) override;

    MAT* InitWeights(int inputSize, int outputSize) override;
};

class LeakyReLU : public Activation
{
public:
    explicit LeakyReLU(double alpha);

#if not USE_GPU

    double Function(double input) override;

#endif

    double Derive(double input) override;

    MAT* InitWeights(int inputSize, int outputSize) override;

    void Save(std::ofstream& writer) override;

private:
    double alpha;
};


class Softmax : public Activation
{
public:
    Softmax();

    void FeedForward(const MAT* input, MAT* output) override;

#if USE_GPU

    void Derivative(const MAT* input, const MAT* lastDelta, const MAT* z, MAT* output) override;

#else

    void Derivative(const MAT* input, MAT* output) override;

    double inline Function(double input) override
    { return 0; };

#endif

    double inline Derive(double input) override
    { return 0; };

    MAT* InitWeights(int inputSize, int outputSize) override;

};

class Tanh : public Activation
{
public:
    Tanh();

#if not USE_GPU

    double Function(double input) override;

#endif

    double Derive(double input) override;

    MAT* InitWeights(int inputSize, int outputSize) override;
};

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

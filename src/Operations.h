#pragma once

class Operation
{
public:
    virtual void Compute() = 0;
};


class Add : public Operation
{
public:
    Add(float* a, float* b,float* output, unsigned int length);
    void Compute() override;
private:
    float* a;
    float* b;
    float* out;
    unsigned int length;
};

class H_Add : public Operation
{
public:
    H_Add(float* a, float* b, int length);
    void Compute() override;
};

class MulAddTo1 : public Operation
{
public:
    MulAddTo1(float* a, float* b,float* output, unsigned int length);
    void Compute() override;
private:
    float* a;
    float* b;
    float* output;
    unsigned int length;
};

class EqualTo : public Operation
{
public:
    EqualTo(float* a, float num, unsigned int length);
    void Compute() override;
private:
    float* a;
    float number;
    unsigned int length;
};


class Mul : public Operation
{
public:
    Mul(float* a, float* b,float* out, unsigned int length);
    void Compute() override;
private:
    float* a;
    float* b;
    float* out;
    unsigned int length;
};

class Sub : public Operation
{
public:
    Sub(float* a, float* b, unsigned int length);
    void Compute() override;

private:
    float* a;
    float* b;
    unsigned int length;
};
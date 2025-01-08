#pragma once


class ActivationTests
{
public:
    static bool ExecuteTests();
private:
    static bool TestReLU();
    static bool TestLeakyReLU();
    static bool TestTanh();
    static bool TestSoftmax();
    static bool TestSigmoid();
    static bool TestSigmoidPrime();
};
// #pragma once
// #include <matrix/Matrix.cuh>
// #include <network/InitFunc.cuh>
// #include "network/activation/Activation.cuh"
// class LeakyReLU
// {
// public:
//     explicit LeakyReLU(double alpha);
//
// #if not USE_GPU
//
//     static double Function(double input);
//
// #endif
//     static double Derive(double input);
//
//     static MAT* InitWeights(int inputSize, int outputSize);
//
//     static void Save(std::ofstream& writer);
//
//     static void FeedForward(const MAT* input, MAT* output)
//     {
// 		DefaultFeedForward(input, output, Function);
//     }
//
//     static void Derivative(const MAT* input, MAT* output)
//     {
//         DefaultDerivative(input, output, Derive);
//     }
//
// private:
//     double alpha;
// };
//
//
//
// LeakyReLU::LeakyReLU(const double _alpha)
// {
//     alpha = _alpha;
// #if USE_GPU
//     throw std::runtime_error("LeakyReLU is not implemented on GPU");
// #endif
// }
//
// #if not USE_GPU
//
// double LeakyReLU::Function(const double input)
// {
//     return input > 0 ? input : 0.01 * input;
// }
//
// #endif
//
// double LeakyReLU::Derive(const double input)
// {
//     return input > 0 ? 1 : 0.01;
// }
//
// void LeakyReLU::Save(std::ofstream& writer)
// {
//     writer.write(reinterpret_cast<const char*>(&ActivationID<LeakyReLU>::value), sizeof(int));
//     writer.write(reinterpret_cast<char*>(&alpha), sizeof(float));
// }
//
// MAT* LeakyReLU::InitWeights(const int previousNeuronsCount, const int NeuronsCount)
// {
// #if USE_GPU
//     auto* weights = new Matrix_GPU(NeuronsCount, previousNeuronsCount);
// #else
//     auto* weights = new Matrix(NeuronsCount, previousNeuronsCount, 1, true);
// #endif
//     WeightsInit::HeUniform(previousNeuronsCount, weights);
//     return weights;
// }
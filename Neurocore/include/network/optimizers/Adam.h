#pragma once
#include "Optimizer.h"

template<double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double gamma = 1e-7>
struct AdamConfig {
    static inline double adjBeta1 = 1.0;
    static inline double adjBeta2 = 1.0;
    static inline double* momentum1 = nullptr;
    static inline double* momentum2 = nullptr;
    static inline double* biasCorrectedMomentum1 = nullptr;
    static inline double* biasCorrectedMomentum2 = nullptr;
};

template<double alpha, double beta1, double beta2, double gamma>
class Optimizer<AdamConfig<alpha, beta1, beta2, gamma>> final
{
public:
    using State = AdamConfig<alpha, beta1, beta2, gamma>;

    static void Compile(int size) {
        if (State::momentum1 == nullptr) {
#if USE_GPU
            checkCUDA(cudaMalloc(&State::momentum1, size * sizeof(double)));
#else
            State::momentum1 = new double[size]();
#endif
        }
        if (State::momentum2 == nullptr) {
#if USE_GPU
            checkCUDA(cudaMalloc(&State::momentum2, size * sizeof(double)));
#else
            State::momentum2 = new double[size]();
#endif
        }
        if (State::biasCorrectedMomentum1 == nullptr) {
#if USE_GPU
            checkCUDA(cudaMalloc(&State::biasCorrectedMomentum1, size * sizeof(double)));
#else
            State::biasCorrectedMomentum1 = new double[size]();
#endif
        }
        if (State::biasCorrectedMomentum2 == nullptr) {
#if USE_GPU
            checkCUDA(cudaMalloc(&State::biasCorrectedMomentum2, size * sizeof(double)));
#else
            State::biasCorrectedMomentum2 = new double[size]();
#endif
        }
    }

    static void Compute(MAT* gradient, MAT* parameters, int offset = 0) {
        double* _momentum1 = State::momentum1 + offset;
        double* _momentum2 = State::momentum2 + offset;
        double* _biasCorrectedMomentum1 = State::biasCorrectedMomentum1 + offset;
        double* _biasCorrectedMomentum2 = State::biasCorrectedMomentum2 + offset;

#if USE_GPU
        const int numBlocks = (gradient->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
        AdamComputeKernel<<<numBlocks, Matrix_GPU::cuda->threadsPerBlock>>>(gradient->GetData() + offset, parameters->GetData() + offset, _momentum1, _momentum2, _biasCorrectedMomentum1, _biasCorrectedMomentum2, gradient->GetSize(), alpha, gamma, beta1, beta2, State::adjBeta1, State::adjBeta2);
#else
        for (int i = 0; i < gradient->GetSize(); i++) {
            const double g = (*gradient)[i];
            _momentum1[i] = beta1 * _momentum1[i] + (1 - beta1) * g;
            _momentum2[i] = beta2 * _momentum2[i] + (1 - beta2) * g * g;
            _biasCorrectedMomentum1[i] = _momentum1[i] / (1 - State::adjBeta1);
            _biasCorrectedMomentum2[i] = _momentum2[i] / (1 - State::adjBeta2);
            (*parameters)[i] = (*parameters)[i] - alpha * _biasCorrectedMomentum1[i] / (sqrt(_biasCorrectedMomentum2[i]) + gamma);
        }
#endif
        State::adjBeta1 *= beta1;
        State::adjBeta2 *= beta2;
    }

    static void cleanup() {
#if USE_GPU
        if (State::momentum1) checkCUDA(cudaFree(State::momentum1));
        if (State::momentum2) checkCUDA(cudaFree(State::momentum2));
        if (State::biasCorrectedMomentum1) checkCUDA(cudaFree(State::biasCorrectedMomentum1));
        if (State::biasCorrectedMomentum2) checkCUDA(cudaFree(State::biasCorrectedMomentum2));
#else
        delete[] State::momentum1;
        delete[] State::momentum2;
        delete[] State::biasCorrectedMomentum1;
        delete[] State::biasCorrectedMomentum2;
#endif
    }
};
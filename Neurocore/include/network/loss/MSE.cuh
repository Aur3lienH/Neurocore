#pragma once
#include "matrix/Matrix.cuh"
#include <cmath>

template<int rows, int cols, int dims>
class MSE final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;
public:
    static double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target) {
#if USE_GPU
        // Allocation de la mémoire CPU pour les calculs
        Matrix outputCPU(output->GetRows(), output->GetCols(), output->GetDims());
        Matrix targetCPU(target->GetRows(), target->GetCols(), target->GetDims());

        // Copie des données du GPU vers le CPU
        checkCUDA(cudaMemcpy(outputCPU.GetData(), output->GetData(),
                            output->GetSize() * sizeof(float), cudaMemcpyDeviceToHost));
        checkCUDA(cudaMemcpy(targetCPU.GetData(), target->GetData(),
                            target->GetSize() * sizeof(float), cudaMemcpyDeviceToHost));

        // Calcul du MSE sur CPU
        double cost = 0.0;
        const int totalSize = output->GetRows() * output->GetCols();

        #pragma omp parallel for reduction(+:cost)
        for (int i = 0; i < totalSize; i++) {
            const double diff = outputCPU[i] - targetCPU[i];
            cost += diff * diff;
        }
#else
        // Version CPU directe
        double cost = 0.0;
        const int totalSize = output->GetRows() * output->GetCols();

        #pragma omp parallel for reduction(+:cost)
        for (int i = 0; i < totalSize; i++) {
            const double diff = output[0][i] - target[0][i];
            cost += diff * diff;
        }
#endif
        // Division par 2*N pour obtenir la moyenne
        return cost / (2.0 * output->GetRows());
    }

    static void CostDerivative(const MAT<rows,cols,dims>* output,
                              const MAT<rows,cols,dims>* target,
                              MAT<rows,cols,dims>* result) {
#if USE_GPU
        // Utilisation d'un kernel CUDA pour le calcul de la dérivée
        const int blocksPerGrid = (output->GetSize() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        MSEDerivativeKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            output->GetData(),
            target->GetData(),
            result->GetData(),
            output->GetSize()
        );

        // Synchronisation pour s'assurer que le kernel est terminé
        checkCUDA(cudaDeviceSynchronize());
#else
        // Version CPU avec potentielle vectorisation
        const int totalSize = output->GetRows() * output->GetCols();

        #pragma omp parallel for
        for (int i = 0; i < totalSize; i++) {
            result[0][i] = output[0][i] - target[0][i];
        }
#endif
    }

private:
#if USE_GPU
    // Kernel CUDA pour le calcul de la dérivée du MSE
    __global__
    static void MSEDerivativeKernel(const float* output, const float* target,
                                   float* result, const int size) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            result[idx] = output[idx] - target[idx];
        }
    }

    static constexpr int THREADS_PER_BLOCK = 256;
#endif
};
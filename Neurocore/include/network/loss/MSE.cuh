#pragma once
#include "matrix/Matrix.cuh"
#include <cmath>

template <int rows, int cols, int dims, bool GPU=GPU_DEFAULT>
class MSE final
{
public:
    static constexpr int Rows = rows;
    static constexpr int Cols = cols;
    static constexpr int Dims = dims;

public:
    static double Cost(const MAT<rows, cols, dims>* output, const MAT<rows, cols, dims>* target)
    {
            double cost = 0.0;
        if constexpr (GPU)
        {
            auto* outputCPU = output->CPU_copy();
            auto* targetCPU = target->CPU_copy();

            // Calcul du MSE sur CPU
            const int totalSize = output->GetRows() * output->GetCols();

#pragma omp parallel for reduction(+:cost)
            for (int i = 0; i < totalSize; i++)
            {
                const double diff = outputCPU->data[i] - targetCPU->data[i];
                cost += diff * diff;
            }

            delete outputCPU;
            delete targetCPU;
        }
        else
        {
            // Version CPU directe
            const int totalSize = output->GetRows() * output->GetCols();

#pragma omp parallel for reduction(+:cost)
            for (int i = 0; i < totalSize; i++) {
                const double diff = output->data[i] - target->data[i];
                cost += diff * diff;
            }
        }
        // Division par 2*N pour obtenir la moyenne
        return cost / (2.0 * output->GetRows());
    }

    static void CostDerivative(const MAT<rows, cols, dims>* output,
                               const MAT<rows, cols, dims>* target,
                               MAT<rows, cols, dims>* result)
    {
        if constexpr (GPU)
        {
            // Utilisation d'un kernel CUDA pour le calcul de la dérivée
            const int blocksPerGrid = (output->GetSize() + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock;

            checkKernel((MSEDerivativeKernel<<<blocksPerGrid, cuda->threadsPerBlock>>>(
                output->GetData(),
                target->GetData(),
                result->GetData(),
                output->GetSize()
            )));
        }
        else
        {
            // Version CPU avec potentielle vectorisation
            const int totalSize = output->GetRows() * output->GetCols();

#pragma omp parallel for
            for (int i = 0; i < totalSize; i++)
            {
                result->data[i] = output->data[i] - target->data[i];
            }
        }
    }
};
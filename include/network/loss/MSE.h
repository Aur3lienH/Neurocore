#pragma once
#include "network/loss/MSE.h"
#include "matrix/Matrix.cuh"


class MSE
{
public:
    MSE();

    template<int rows, int cols, int dims>
    double Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target);

    template<int rows, int cols, int dims>
    void CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result);
};


template<int rows, int cols, int dims>
double MSE::Cost(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target)
{
    double cost = 0;
#if USE_GPU
    std::cout << "MSE::Cost kernel not implemented yet\n";
    Matrix outputCPU(output->GetRows(), output->GetCols(), output->GetDims());
    checkCUDA(cudaMemcpy(outputCPU.GetData(), output->GetData(), output->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    Matrix targetCPU(target->GetRows(), target->GetCols(), target->GetDims());
    checkCUDA(cudaMemcpy(targetCPU.GetData(), target->GetData(), target->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
#endif
    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
    {
#if USE_GPU
        cost += pow(outputCPU[i] - targetCPU[i], 2);
#else
        cost += pow(output[0][i] - target[0][i], 2);
#endif
    }
    return cost / (2 * output->GetRows());
}

template<int rows, int cols, int dims>
void MSE::CostDerivative(const MAT<rows,cols,dims>* output, const MAT<rows,cols,dims>* target, MAT<rows,cols,dims>* result)
{
#if USE_GPU
    std::cout << "MSE::CostDerivative kernel not implemented yet\n";
    Matrix outputCPU(output->GetRows(), output->GetCols(), output->GetDims());
    checkCUDA(cudaMemcpy(outputCPU.GetData(), output->GetData(), output->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    Matrix targetCPU(target->GetRows(), target->GetCols(), target->GetDims());
    checkCUDA(cudaMemcpy(targetCPU.GetData(), target->GetData(), target->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    Matrix resultCPU(result->GetRows(), result->GetCols(), result->GetDims());
    checkCUDA(cudaMemcpy(resultCPU.GetData(), result->GetData(), result->GetSize() * sizeof(float),
                         cudaMemcpyDeviceToHost));
#endif
    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
    {
#if USE_GPU
        resultCPU[i] = outputCPU[i] - targetCPU[i];
#else
        result[0][i] = output[0][i] - target[0][i];
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(result->GetData(), resultCPU.GetData(), result->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
#endif
}


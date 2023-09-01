#include "Loss.cuh"
#include <cmath>
#include "Matrix.cuh"


Loss::Loss()
{
}

void Loss::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&ID), sizeof(int));
}

Loss* Loss::Read(std::ifstream& reader)
{
    int id;
    reader.read(reinterpret_cast<char*>(&id), sizeof(int));
    if (id == 0)
    {
        return new MSE();
    }
    else if (id == 1)
    {
        return new CrossEntropy();
    }
    else
    {
        throw std::invalid_argument("Invalid ID : Loss function");
    }
}


MSE::MSE()
{
    ID = 0;
}

double MSE::Cost(const MAT* output, const MAT* target)
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

void MSE::CostDerivative(const MAT* output, const MAT* target, MAT* result)
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


CrossEntropy::CrossEntropy()
{
    ID = 1;
}

__global__
void CrossEntropyCostKernel(double* cost, double* output, double* target, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        cost[i] = target[i] * log(output[i] + 1e-15) + (1 - target[i]) * log(1 - output[i] + 1e-15);
    }
}

double CrossEntropy::Cost(const MAT* output, const MAT* target)
{
#if USE_GPU
    Matrix outputCPU(output->GetRows(), output->GetCols(), output->GetData_CPU_1D());
    Matrix targetCPU(target->GetRows(), target->GetCols(), target->GetData_CPU_1D());
    double cost = 0;

    for (int i = 0; i < output->GetMatrixSize(); i++)
        cost += targetCPU[i] * log(outputCPU[i] + 1e-15) + (1 - targetCPU[i]) * log(1 - outputCPU[i] + 1e-15);

    return -cost / output->GetRows();
#else
    double cost = 0;

    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
        cost += target[0][i] * log(output[0][i] + 1e-15) + (1 - target[0][i]) * log(1 - output[0][i] + 1e-15);

    return -cost / output->GetRows();
#endif
}

__global__
void CostDerivativeKernel(const float* output, const float* target, float* result, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        if (target[i] == 1)
        {
            result[i] = -1 + output[i];
        }
        else
        {
            result[i] = output[i];
        }
    }
}

void CrossEntropy::CostDerivative(const MAT* output, const MAT* target, MAT* result)
{
#if USE_GPU
    const int blocksPerGrid =
            (output->GetSize() + Matrix_GPU::cuda->threadsPerBlock - 1) / Matrix_GPU::cuda->threadsPerBlock;
    CostDerivativeKernel<<<blocksPerGrid, Matrix_GPU::cuda->threadsPerBlock>>>(output->GetData(), target->GetData(),
                                                                               result->GetData(), output->GetSize());
    checkCUDA(cudaDeviceSynchronize());
#else
    for (int i = 0; i < output->GetRows() * output->GetCols(); i++)
    {
        if (target[0][i] == 1)
        {
            result[0][i] = -1 + output[0][i];
        }
        else
        {
            result[0][i] = output[0][i];
        }
    }
#endif
}






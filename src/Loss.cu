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

double CrossEntropy::Cost(const MAT* output, const MAT* target)
{
    double cost = 0;
#if USE_GPU
    std::cout << "CrossEntropy::Cost kernel not implemented yet\n";
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
        cost += targetCPU[i] * log(outputCPU[i] + 1e-15) + (1 - targetCPU[i]) * log(1 - outputCPU[i] + 1e-15);
#else
        cost += target[0][i] * log(output[0][i] + 1e-15) + (1 - target[0][i]) * log(1 - output[0][i] + 1e-15);
#endif
    }
    return -cost / output->GetRows();
}

void CrossEntropy::CostDerivative(const MAT* output, const MAT* target, MAT* result)
{
#if USE_GPU
    std::cout << "CrossEntropy::CostDerivative kernel not implemented yet\n";
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
        if (targetCPU[i] == 1)
        {
            resultCPU[i] = -1 / outputCPU[i];
        }
        else
        {
            resultCPU[i] = outputCPU[i];
        }
#else
        if (target[0][i] == 1)
        {
            result[0][i] = -1 + output[0][i];
        }
        else
        {
            result[0][i] = output[0][i];
        }
#endif
    }

#if USE_GPU
    checkCUDA(cudaMemcpy(result->GetData(), resultCPU.GetData(), result->GetSize() * sizeof(float),
                         cudaMemcpyHostToDevice));
#endif
}






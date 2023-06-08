#include "Loss.h"
#include "math.h"
#include "Matrix.h"


Loss::Loss()
{
}

double MSE::Cost(const Matrix* output, const Matrix* target)
{
    double cost = 0;
    for (int i = 0; i < output->getRows() * output->getCols(); i++)
    {
        cost += pow(output[0][i] - target[0][i], 2);
    }
    return cost / (2 * output->getRows());
}

void MSE::CostDerivative(const Matrix* output, const Matrix* target, Matrix* result)
{
    for (int i = 0; i < output->getRows() * output->getCols(); i++)
    {
        result[0][i] = output[0][i] - target[0][i];
    }
}


double CrossEntropy::Cost(const Matrix* output, const Matrix* target)
{
    double cost = 0;
    for (int i = 0; i < output->getRows() * output->getCols(); i++)
    {
        cost += target[0][i] * log(output[0][i]) + (1 - target[0][i]) * log(1 - output[0][i]);
    }
    return -cost / output->getRows();
}

void CrossEntropy::CostDerivative(const Matrix* output, const Matrix* target, Matrix* result)
{
    for (int i = 0; i < output->getRows() * output->getCols(); i++)
    {
        if(target[0][i] == 1)
        {
            result[0][i] = -1 + output[0][i]; 
        }
        else
        {
            result[0][i] = output[0][i]; 
        }
    }
}


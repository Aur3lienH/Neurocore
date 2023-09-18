#include "Matrix.cuh"
#include "Operations.h"
#include <iostream>
#include <vector>

/*
std::vector<Operation*> Matrix::O_CrossProduct(Matrix* a, Matrix* b, Matrix* output)
{
    std::vector<Operation*> res;
    res.push_back(new EqualTo(output->GetData(),0,output->size()));
    if(!a->IsColumnMajor() && b->IsColumnMajor())
    {
        for (int i = 0; i < a->getRows(); i++)
        {
            for (int j = 0; j < a->getCols(); j++)
            {
                res.push_back(new MulAddTo1(a->GetData() + i * a->getCols(),b->GetData() + j * a->getCols(),output->GetData() + i * a->getCols() + j,a->getCols()));
            }   
        }
    }
    else
    {
        throw std::invalid_argument("not implemented !\n");
    }

    return res;
}
*/
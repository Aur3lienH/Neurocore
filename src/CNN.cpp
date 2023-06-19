#include <iostream>
#include <vector>
#include "CNN.h"


CNN::CNN(int* _sizes) : Layer(_sizes,2)
{
    sizes = _sizes;
}

Matrix* CNN::FeedForward(const Matrix* inputs)
{
    for (int j = configRow[0]; j < configRow[1]; j++)
    {
        for (int i = configCol[0]; i < configCol[1]; i++)
        {
            
            for (int k = -filters->getRows() / 2; k < filters->getRows() / 2 + 1; k++)
            {
                double res = 0;
                for (int l = -filters->getCols() / 2 ; l < filters->getCols() / 2 + 1; l++)
                {
                    res += inputs->operator()(i + k,j + l) * filters->operator()(l,k);
                }
                
            }
            
        }
    }
    
    
}


//
// Created by matmu on 20/06/2023.
//

#include "ConvLayer.h"

ConvLayer::ConvLayer(MatrixCarre* _filter) : filter(_filter)
{

}

void ConvLayer::Convolve(Matrix* input, Matrix* output)
{
    int filterSize = filter->getRows();
    int inputCols = input->getCols();
    int inputRows = input->getRows();
    int outputCols = inputCols - filterSize + 1;
    int outputRows = inputRows - filterSize + 1;

    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            double sum = 0;
            for (int k = 0; k < filterSize; k++)
            {
                for (int l = 0; l < filterSize; l++)
                {
                    sum += (*input)(i + k, j + l) * (*filter)(k, l);
                }
            }
            (*output)(i, j) = sum;
        }
    }
}

void ConvLayer::FullConvolve(Matrix* input, Matrix* output)
{
    int filterSize = filter->getRows();
    int inputCols = input->getCols();
    int inputRows = input->getRows();
    int outputCols = inputCols + filterSize - 1;
    int outputRows = inputRows + filterSize - 1;

    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            double sum = 0;
            for (int k = 0; k < filterSize; k++)
            {
                for (int l = 0; l < filterSize; l++)
                {
                    if (i - k >= 0 && i - k < inputRows && j - l >= 0 && j - l < inputCols)
                    {
                        sum += (*input)(i - k, j - l) * (*filter)(k, l);
                    }
                }
            }
            (*output)(i, j) = sum;
        }
    }
}

#include "ImagePreProcessing.cuh"
#include "Matrix.cuh"
#include <cmath>
#include <random>


std::default_random_engine ImagePreProcessing::generator;
std::uniform_real_distribution<double> ImagePreProcessing::distribution(0.0,1.0);


double ImagePreProcessing::Random(double mean, double stddev)
{
    // Use Box-Muller transform to generate a point from a Gaussian distribution
    double u = distribution(generator);
    double v = distribution(generator);
    double x = std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * M_PI * v);
    x = x * stddev + mean;
    return x;
}


Matrix* ImagePreProcessing::ImageWithNoise(const Matrix* origianl)
{
    Matrix* res = new Matrix(origianl->getRows(),origianl->getCols(),origianl->getDim());


    for (int k = 0; k < origianl->getDim(); k++)
    {
        for (int i = 0; i < origianl->getRows(); i++)
        {
            for (int j = 0; j < origianl->getCols(); j++)
            {
                (*res)(i,j) = (*origianl)(i,j) + Random(10,0);
            }
        }
    }
    return res;
}

Matrix* ImagePreProcessing::Rotate(const Matrix* original, double angle)
{
    Matrix* res = new Matrix(original->getRows(),original->getCols(),original->getDim());

    for (int k = 0; k < original->getDim(); k++)
    {
        for (int i = 0; i < original->getRows(); i++)
        {
            for (int j = 0; j < original->getCols(); j++)
            {
                double x = i - original->getRows() / 2.0;
                double y = j - original->getCols() / 2.0;
                double x1 = x * std::cos(angle) - y * std::sin(angle);
                double y1 = x * std::sin(angle) + y * std::cos(angle);
                x1 += original->getRows() / 2.0;
                y1 += original->getCols() / 2.0;
                if (x1 < 0 || x1 >= original->getRows() || y1 < 0 || y1 >= original->getCols())
                    continue;
                (*res)(i,j) = (*original)(x1,y1);
            }
        }
        original->GoToNextMatrix();
        res->GoToNextMatrix();
    }
    original->ResetOffset();
    res->ResetOffset();
    return res;
}

Matrix* ImagePreProcessing::DownSize(Matrix* input, int rows, int cols)
{
    Matrix* res = new Matrix(rows,cols,input->getDim());

    for (int k = 0; k < input->getDim(); k++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                (*res)(i,j) = (*input)(i * input->getRows() / rows,j * input->getCols() / cols);
            }
        }
        input->GoToNextMatrix();
        res->GoToNextMatrix();
    }
    input->ResetOffset();
    res->ResetOffset();
    return res;
}


Matrix* ImagePreProcessing::ToBlackAndWhite(Matrix** original, bool alpha)
{
    Matrix* res = new Matrix((*original)->getRows(),(*original)->getCols(),1);

    for (int k = 0; k < (*original)->getDim(); k++)
    {
        for (int i = 0; i < (*original)->getRows(); i++)
        {
            for (int j = 0; j < (*original)->getCols(); j++)
            {
                if (alpha)
                {
                    (*res)(i,j) = 0.299 * (*(*original))(i,j) + 0.587 * (*(*original))(i,j) + 0.114 * (*(*original))(i,j);
                }
                else
                {
                    (*res)(i,j) = 0.299 * (*(*original))(i,j) + 0.587 * (*(*original))(i,j) + 0.114 * (*(*original))(i,j);
                }
            }
        }
        (*original)->GoToNextMatrix();
        res->GoToNextMatrix();
    }
    (*original)->ResetOffset();
    res->ResetOffset();


    return res;
}


Matrix* ImagePreProcessing::Stretch(const Matrix* input, double x, double y)
{
    Matrix* res = new Matrix(input->getRows() * x,input->getCols() * y,input->getDim());

    for (int k = 0; k < input->getDim(); k++)
    {
        for (int i = 0; i < input->getRows() * x; i++)
        {
            for (int j = 0; j < input->getCols() * y; j++)
            {
                (*res)(i,j) = (*input)(i / x,j / y);
            }
        }
        input->GoToNextMatrix();
        res->GoToNextMatrix();
    }
    input->ResetOffset();
    res->ResetOffset();
    return res;
}




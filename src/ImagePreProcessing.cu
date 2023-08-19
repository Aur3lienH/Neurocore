#include "ImagePreProcessing.cuh"
#include "Matrix.cuh"
#include <cmath>
#include <random>


std::default_random_engine ImagePreProcessing::generator;
std::uniform_real_distribution<double> ImagePreProcessing::distribution(0.0, 1.0);


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
    Matrix* res = new Matrix(origianl->GetRows(), origianl->GetCols(), origianl->GetDims());


    for (int k = 0; k < origianl->GetDims(); k++)
    {
        for (int i = 0; i < origianl->GetRows(); i++)
        {
            for (int j = 0; j < origianl->GetCols(); j++)
            {
                (*res)(i, j) = (*origianl)(i, j) + Random(10, 0);
            }
        }
    }
    return res;
}

Matrix* ImagePreProcessing::Rotate(const Matrix* original, double angle)
{
    Matrix* res = new Matrix(original->GetRows(), original->GetCols(), original->GetDims());

    for (int k = 0; k < original->GetDims(); k++)
    {
        for (int i = 0; i < original->GetRows(); i++)
        {
            for (int j = 0; j < original->GetCols(); j++)
            {
                double x = i - original->GetRows() / 2.0;
                double y = j - original->GetCols() / 2.0;
                double x1 = x * std::cos(angle) - y * std::sin(angle);
                double y1 = x * std::sin(angle) + y * std::cos(angle);
                x1 += original->GetRows() / 2.0;
                y1 += original->GetCols() / 2.0;
                if (x1 < 0 || x1 >= original->GetRows() || y1 < 0 || y1 >= original->GetCols())
                    continue;
                (*res)(i, j) = (*original)(x1, y1);
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
    Matrix* res = new Matrix(rows, cols, input->GetDims());

    for (int k = 0; k < input->GetDims(); k++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                (*res)(i, j) = (*input)(i * input->GetRows() / rows, j * input->GetCols() / cols);
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
    Matrix* res = new Matrix((*original)->GetRows(), (*original)->GetCols(), 1);

    for (int k = 0; k < (*original)->GetDims(); k++)
    {
        for (int i = 0; i < (*original)->GetRows(); i++)
        {
            for (int j = 0; j < (*original)->GetCols(); j++)
            {
                if (alpha)
                {
                    (*res)(i, j) =
                            0.299 * (*(*original))(i, j) + 0.587 * (*(*original))(i, j) + 0.114 * (*(*original))(i, j);
                }
                else
                {
                    (*res)(i, j) =
                            0.299 * (*(*original))(i, j) + 0.587 * (*(*original))(i, j) + 0.114 * (*(*original))(i, j);
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
    Matrix* res = new Matrix(input->GetRows() * x, input->GetCols() * y, input->GetDims());

    for (int k = 0; k < input->GetDims(); k++)
    {
        for (int i = 0; i < input->GetRows() * x; i++)
        {
            for (int j = 0; j < input->GetCols() * y; j++)
            {
                (*res)(i, j) = (*input)(i / x, j / y);
            }
        }
        input->GoToNextMatrix();
        res->GoToNextMatrix();
    }
    input->ResetOffset();
    res->ResetOffset();
    return res;
}




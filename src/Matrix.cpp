#include <iostream>
#include <fstream>
#include <cmath>
#include <float.h>
#include "Matrix.h"


#define SAFE true

//MATRIX

Matrix::Matrix()
{

}


Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    matrixSize = rows * cols;
    this->data = new double[rows * cols * dim];

    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = 0;
    }
}

Matrix::Matrix(int rows, int cols, int dim)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dim;  
    matrixSize = rows * cols;
    this->data = new double[rows * cols * dim];
}


Matrix::Matrix(int rows, int cols, double value)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    matrixSize = rows * cols;
    this->data = new double[rows * cols * dim];
    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = value;
    }
}

Matrix::Matrix(int rows, int cols, double* newArray)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    this->data = newArray;
    matrixSize = rows * cols;
}

Matrix::Matrix(int rows, int cols, int dims, double* data)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dims;
    this->data = data;
    matrixSize = rows * cols;
}

Matrix::~Matrix()
{
    delete[] this->data;
}

const int Matrix::getRows () const
{
    return this->rows;
}

const int Matrix::getCols() const
{
    return this->cols;
}

const int Matrix::getDim() const
{
    return this->dim;
}

void Matrix::Add(Matrix* other, Matrix* result)
{
    
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        std::cout << "Matrix 1: " << this->rows << "x" << this->cols << std::endl;
        std::cout << "Matrix 2: " << other->rows << "x" << other->cols << std::endl;
        return;
    }
#endif
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        result->data[i] = this->data[i] + other->data[i];
    }
}

void Matrix::AddAllDims(Matrix* other, Matrix* result)
{
    #if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    int size = this->rows * this->cols * this->dim;
    
    for (int i = 0; i < size; i++)
    {
        result->data[i] = this->data[i] + other->data[i];
    }
}

void Matrix::Substract(const Matrix* other, Matrix* result) const 
{
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        result->data[i] = this->data[i] - other->data[i];
    }
}

void Matrix::SubstractAllDims(const Matrix* other, Matrix* result) const
{
    #if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    int size = this->rows * this->cols * this->dim;
    
    for (int i = 0; i < size; i++)
    {
        result->data[i] = this->data[i] - other->data[i];
    }
}


void Matrix::MultiplyAllDims(const Matrix* other, Matrix* result) const
{
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    int size = this->rows * this->cols * this->dim;
    
    for (int i = 0; i < size; i++)
    {
        result->data[i] = this->data[i] * other->data[i];
    }
}

void Matrix::MultiplyAllDims(double value)
{
    int size = this->rows * this->cols * this->dim;
    
    for (int i = 0; i < size; i++)
    {
        this->data[i] *= value;
    }
}


void Matrix::DivideAllDims(double value)
{
    int size = this->rows * this->cols * this->dim;
    
    for (int i = 0; i < size; i++)
    {
        this->data[i] /= value;
    }
    
}

void Matrix::Zero()
{
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        this->data[i] = 0;
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
    std::cout << "Matrix: " << matrix.rows << "x" << matrix.cols << std::endl;
    for (int i = 0; i < matrix.rows; i++)
    {
        os << "[";
        for (int j = 0; j < matrix.cols; j++)
        {
            os << matrix.data[i * matrix.cols + j];
            if(j != matrix.cols - 1)
            {
                os << " ";
            }
        }
        os << "]\n";
    }
    return os;
}

void Matrix::PrintSize() const
{
    std::cout << "(" << this->rows << "," << this->cols << "," << this->dim  << ")" << std::endl;
}

double& Matrix::operator[](int index) 
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return this->data[index];
    
}

const double& Matrix::operator[](int index) const
{
    
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return this->data[index];
}

const double& Matrix::operator()(int _rows, int _cols) const
{
#if SAFE
    if(_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    return data[_rows * this->cols + _cols];
}

Matrix* Matrix::operator*=(const Matrix* other)
{
#if SAFE
    if(this->cols != other->cols && this->rows != other->rows)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
#endif

    for (int i = 0; i < cols*rows; i++)
    {
        this->data[i] *= other->data[i];
    }
    return this;
}

Matrix* Matrix::operator*=(const double other)
{
    for (int i = 0; i < cols*rows; i++)
    {
        this->data[i] = data[i] * other;
    }
    return this;
}




Matrix* Matrix::operator+(const Matrix& other)
{
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    Matrix* result = new Matrix(this->rows, this->cols);
    for (int i = 0; i < this->cols * this->rows; i++)
    {
        
        result->operator[](i) += other.data[i];
        
    }
    return result;
}
Matrix* Matrix::operator+=(const Matrix& other)
{
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif
    
    for (int i = 0; i < this->cols * this->rows; i++)
    {
        
        this->data[i] += other.data[i];
        
    }
    return this;
}


Matrix* Matrix::operator*(const double& other)
{
    for (int i = 0; i < this->cols * this->rows; i++)
    {
        
        this->data[i] *= other;
        
    }
    return this;
}

Matrix* Matrix::operator-=(const Matrix& other)
{
#if SAFE
    if(this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        
        this->data[i] -= other.data[i];
        
    }
    return this;
}

const void Matrix::CrossProduct(const Matrix* other, Matrix* output) const
{
#if SAFE
    
    if(other->rows != this->cols)
    {
        throw std::runtime_error("Matrice have not the shape to be cross producted !");
        return;
    }
    if(output->rows != this->rows || output->cols != other->cols)
    {
        throw std::runtime_error("Output matrix has not the right shape !");
        return;
    }

#endif
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < other->cols; j++)
        {
            output->data[i * other->cols + j] = 0;
            for (int k = 0; k < this->cols; k++)
            {
                output->data[i * output->cols + j] += this->data[i * this->cols + k] * other->data[k * other->cols + j];
            }
        }
    }
}


//Read and write matrices.

Matrix* Matrix::Read(std::ifstream& reader)
{
    int row,col,dim;
    reader.read(reinterpret_cast<char*>(&row),sizeof(int));
    reader.read(reinterpret_cast<char*>(&col),sizeof(int));
    reader.read(reinterpret_cast<char*>(&dim),sizeof(int));
    Matrix* matrix = new Matrix(row,col,dim);
    for (int i = 0; i < row * col * dim; i++)
    {
        reader.read(reinterpret_cast<char*>(matrix->data + i),sizeof(double));
    }
    return matrix;
}

void Matrix::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&rows),sizeof(int));
    writer.write(reinterpret_cast<char*>(&cols),sizeof(int));
    writer.write(reinterpret_cast<char*>(&dim),sizeof(int));
    for (int i = 0; i < rows * cols * dim; i++)
    {
        writer.write(reinterpret_cast<char*>(data + i),sizeof(double));
    }
}

float Matrix::Distance(Matrix* a, Matrix* b)
{
#if SAFE
    if(a->cols != b->cols || a->rows != b->rows)
    {
        throw std::invalid_argument("Matrices need to have same size to calculate distance !");
    }
#endif
    float res = 0;
    for (int i = 0; i < a->cols * a->rows; i++)
    {
        res += (a[0][i] - b[0][i]) * (a[0][i] - b[0][i]);
    }
    res = sqrt(res);
    return res;
}

Matrix* Matrix::Copy()
{
    double* resArray = new double[cols*rows*dim];
    for (int i = 0; i < cols*rows*dim; i++)
    {
        resArray[i] = data[i];
    }
    return new Matrix(rows,cols,dim,resArray);
}

void Matrix::Flip180(const Matrix* input, Matrix* output)
{
    for (int i = 0; i < input->cols / 2; ++i)
    {
        for (int j = 0; j < input->rows / 2; ++j)
        {
            //UGLY
            (*output)(i,j) = (*input)(input->rows - 1 - j,input->cols - 1 - i);
        }
    }
}


const void Matrix::GoToNextMatrix() const
{
    data += matrixSize;
    offset += matrixSize;
}

void Matrix::ResetOffset() const
{
    data -= offset;
    offset = 0;
}

void Matrix::SetOffset(int offset) const
{
    data += offset;
    this->offset += offset;
}

int Matrix::GetOffset() const
{
    return offset;
}


void Matrix::FullConvolution(const Matrix* m, const Matrix* filter, Matrix* output)
{
    const int outputCols = m->getCols() + filter->getCols() - 1;
    const int outputRows = m->getRows() + filter->getRows() - 1;

#if SAFE
    if(output->cols != outputCols || outputRows != output->rows)
    {
        std::cout << "right shape is : " << "(" << outputRows << "," << outputCols << ")\n";
        throw std::invalid_argument("FullConvolution : Output Matrix has not the right shape ! ");
    }
#endif
    const int filterCols = filter->getCols();
    const int filterRows = filter->getRows();

    const int inputCols = m->getCols();
    const int inputRows = m->getRows();

    const int r = filterRows - 1;
    const int c = filterCols - 1;
    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            double sum = 0;
            for (int k = 0; k < filterRows; k++)
            {
                for (int l = 0; l < filterCols; l++)
                {
                    const int inputRow = i + k - r;
                    const int inputCol = j + l - c;
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                    {
                        sum += (*m)(inputRow, inputCol) * (*filter)(k, l);
                    }
                }
            }
            (*output)(i, j) = sum;
        }
    }
}

void Matrix::AveragePool(const Matrix* a, Matrix* output, int filterSize, int stride)
{
    const int inputCols = a->cols;
    const int inputRows = a->rows;
    const int outputCols = (inputCols - filterSize) / stride + 1;
    const int outputRows = (inputRows - filterSize) / stride + 1;

    const int fsSquare = filterSize * filterSize;

    for (int d = 0; d < a->dim; d++)
    {
        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputCols; j++)
            {
                double sum = 0;
                for (int k = 0; k < filterSize; k++)
                {
                    for (int l = 0; l < filterSize; l++)
                    {
                        const int inputRow = i * stride + k;
                        const int inputCol = j * stride + l;
                        if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                            sum += (*a)(inputRow, inputCol);

                    }
                }
                (*output)(i, j) = sum / fsSquare;
            }
        }
        a->GoToNextMatrix();
        output->GoToNextMatrix();
    }

    a->ResetOffset();
    output->ResetOffset();
}


void Matrix::Convolution(const Matrix* input, const Matrix* filter, Matrix* output, int stride)
{


#if SAFE
    int filterSize = filter->getRows();
    int inputCols = input->getCols();
    int inputRows = input->getRows();
    int outputCols = (inputCols - filterSize) / stride + 1;
    int outputRows = (inputRows - filterSize) / stride + 1;
    if(outputCols != output->cols || output->rows != outputRows)
    {
        std::cout << outputRows << "\n";
        throw std::invalid_argument("Convolution : output matrix has not the right shape !");
    }
#endif

    for (int i = 0; i < output->rows; i++)
    {
        for (int j = 0; j < output->cols; j++)
        {
            double sum = 0;
            for (int k = 0; k < filter->getRows(); k++)
            {
                for (int l = 0; l < filter->getRows(); l++)
                {
                    sum += (*input)(i * stride + k, j * stride + l) * (*filter)(k, l);
                }
            }
            (*output)(i, j) = sum;
        }
    }
}

double Matrix::Sum()
{
    double res = 0;
    for (int i = 0; i < cols*rows; i++)
    {
        res += data[i];
    }
    return res;
}

double& Matrix::operator()(int _rows, int _cols)
{
    if(_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
    return data[_rows * this->cols + _cols];
}

void Matrix::Flatten() const
{
    rows *= cols * dim;
    cols = 1;
    dim = 1;
    matrixSize = rows * cols;
}

void Matrix::Reshape(int rows, int cols, int dims) const
{
#if SAFE
    if(rows * cols * dims != this->cols * this->rows * this->dim)
    {
        throw std::invalid_argument("Reshape : Incorrect Reshape !");
    }
#endif

    this->rows = rows;
    this->cols = cols;
    this->dim = dims;
    matrixSize = rows * cols;
}

const int Matrix::size() const
{
    return matrixSize * dim;
}


Matrix* Matrix::Copy(const Matrix* a)
{
    Matrix* res = new Matrix(a->rows,a->cols,a->dim);
    for (int i = 0; i < a->cols * a->rows * a->dim; i++)
    {
        res[0][i] = a[0][i];
    }
    return res;
}

void Matrix::MaxPool(const Matrix* a, Matrix* output, const int filterSize, const int stride)
{
    const int inputCols = a->cols;
    const int inputRows = a->rows;
    const int outputCols = (inputCols - filterSize) / stride + 1;
    const int outputRows = (inputRows - filterSize) / stride + 1;

    for (int d = 0; d < a->dim; d++)
    {
        for (int i = 0; i < outputRows; i++)
        {
            for (int j = 0; j < outputCols; j++)
            {
                double max = -DBL_MAX;
                for (int k = 0; k < filterSize; k++)
                {
                    for (int l = 0; l < filterSize; l++)
                    {
                        const int inputRow = i * stride + k;
                        const int inputCol = j * stride + l;
                        if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                            max = std::max(max, (*a)(inputRow, inputCol));

                    }
                }
                (*output)(i, j) = max;
            }
        }
        a->GoToNextMatrix();
        output->GoToNextMatrix();
    }
    a->ResetOffset();
    output->ResetOffset();
}


Matrix Matrix::Random(int rows, int cols)
{
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        res[i] = (double) rand() / RAND_MAX * 2 - 1;

    return res;
}




//MATRIX CARRE

MatrixCarre::MatrixCarre(int size) : Matrix(size, size)
{
    this->operator[](0) = 1;
}

MatrixCarre::MatrixCarre(int size, double value) : Matrix(size, size)
{
    for (int i = 0; i < size*size; i++)
    {
        this->operator[](i) = value;
    }
}

//MATRIX DIAGONALE

MatrixDiagonale::MatrixDiagonale(int size, double value) : Matrix(size, size)
{
    for (int i = 0; i < size; i++)
    {
        this->operator[](i * size + i) = value; 
    }
}

MatrixDiagonale::~MatrixDiagonale()
{
    
}




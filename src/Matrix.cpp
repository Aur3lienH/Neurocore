#include <iostream>
#include <fstream>
#include <cmath>
#include "Matrix.h"

//MATRIX

Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->data = new double[rows * cols];

    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = 0;
    }
}

Matrix::Matrix(int rows, int cols, double value)
{
    this->rows = rows;
    this->cols = cols;
    this->data = new double[rows * cols];
    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = value;
    }
}

Matrix::Matrix(int rows, int cols, double* newArray)
{
    this->rows = rows;
    this->cols = cols;
    this->data = newArray;
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

void Matrix::Add(Matrix* other, Matrix* result)
{
    
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        std::cout << "Matrix 1: " << this->rows << "x" << this->cols << std::endl;
        std::cout << "Matrix 2: " << other->rows << "x" << other->cols << std::endl;
        return;
    }
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        result->data[i] = this->data[i] + other->data[i];
    }
}

void Matrix::Subtract(const Matrix* other, Matrix* result) const 
{
    
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        result->data[i] = this->data[i] - other->data[i];
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

void Matrix::PrintSize()
{
    std::cout << "(" << this->rows << "," << this->cols << ")" << std::endl;
}

double& Matrix::operator[](int index) 
{
    
    if (index < this->rows * this->cols)
    {
        return this->data[index];
    }
    else
    {
        throw std::out_of_range("Matrix : Index out of bounds");
        return this->data[0];
    }
    
}

const double& Matrix::operator[](int index) const
{
    
    if (index < this->rows * this->cols)
    {
        return this->data[index];
    }
    else
    {
        throw std::out_of_range("Matrix : Index out of bounds");
        return this->data[0];
    }
}

const double& Matrix::operator()(int _rows, int _cols) const
{

    if(_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }

    return data[_rows + _cols * rows];
}

Matrix* Matrix::operator*=(const Matrix* other)
{
    if(this->cols != other->cols && this->rows != other->rows)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
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
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
    Matrix* result = new Matrix(this->rows, this->cols);
    for (int i = 0; i < this->cols * this->rows; i++)
    {
        
        result->operator[](i) += other.data[i];
        
    }
    return result;
}
Matrix* Matrix::operator+=(const Matrix& other)
{
    
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
    
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
    
    if(this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
    
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        
        this->data[i] -= other.data[i];
        
    }
    return this;
}

const void Matrix::CrossProduct(const Matrix* other, Matrix* output) const
{
    
    if(other->rows != this->cols)
    {
        throw std::runtime_error("Matrice have not the shape to be cross producted !");
        return;
    }
    if(output->cols != other->cols || output->rows != this->rows)
    {
        throw std::runtime_error("Output matrix has not the right shape !");
        return;
    }
    
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
    int row,col;
    reader.read(reinterpret_cast<char*>(&row),sizeof(int));
    reader.read(reinterpret_cast<char*>(&col),sizeof(int));
    Matrix* matrix = new Matrix(row,col);
    for (int i = 0; i < row * col; i++)
    {
        reader.read(reinterpret_cast<char*>(matrix->data + i),sizeof(double));
    }
    return matrix;
}

void Matrix::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&rows),sizeof(int));
    writer.write(reinterpret_cast<char*>(&cols),sizeof(int));
    for (int i = 0; i < rows * cols; i++)
    {
        writer.write(reinterpret_cast<char*>(data + i),sizeof(double));
    }
}

float Matrix::Distance(Matrix* a, Matrix* b)
{
    if(a->cols != b->cols || a->rows != b->rows)
    {
        throw std::invalid_argument("Matrices need to have same size to calculate distance !");
    }
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
    double* resArray = new double[cols*rows];
    for (int i = 0; i < cols*rows; i++)
    {
        resArray[i] = data[i];
    }
    return new Matrix(rows,cols,resArray);
}

void MatrixCarre::Flip180()
{
    for (int i = 0; i < this->cols / 2; ++i)
    {
        for (int j = 0; j < this->rows / 2; ++j)
        {
            std::swap(this->data[i * this->cols + j], this->data[(this->cols - i - 1) * this->cols + (this->rows - j - 1)]);
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



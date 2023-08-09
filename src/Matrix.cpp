#include <cmath>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <emmintrin.h>
#include <stdlib.h>
#include "Matrix.h"


#define SAFE false

//MATRIX

Matrix::Matrix()
{

}


Matrix::Matrix(const int rows, const int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    matrixSize = rows * cols;
    this->data = new float[rows * cols * dim];

    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = 0;
    }
}

Matrix::Matrix(const int rows, int cols, int dim, bool aligned)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dim;
    matrixSize = rows * cols;
    if(aligned)
    {
        if(posix_memalign((void**)&data,32,sizeof(float) * rows * cols * dim))
        {
            throw std::invalid_argument("Cannot create an aligned array ! ");
        }
    }
    else
    {
        this->data = new float[rows * cols * dim];
    }
    
    for (int i = 0; i < rows * cols * dim; i++)
    {
        data[i] = 0;
    }





    
    
}


Matrix::Matrix(const int rows, const int cols, float value)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    matrixSize = rows * cols;
    this->data = new float[rows * cols * dim];
    for (int i = 0; i < rows * cols; i++)
    {
        this->data[i] = value;
    }
}

Matrix::Matrix(const int rows, const int cols, float* newArray)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    this->data = newArray;
    matrixSize = rows * cols;
}

Matrix::Matrix(const int rows, const int cols, const int dims, float* data)
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

int Matrix::getRows() const
{
    return this->rows;
}

int Matrix::getCols() const
{
    return this->cols;
}

int Matrix::getDim() const
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
/*
    for (int i = 0; i < this->rows * this->cols; i++)
    {
        result->data[i] = this->data[i] + other->data[i];
    }
*/

    float* temp = new float[4];

    int size = this->rows * this->cols;
    int i;
    for (i = 0; i + 4 <= size; i+=4)
    {
        __m128 sum = _mm_setzero_ps();
        __m128 a = _mm_loadu_ps(data + i);
        __m128 b = _mm_loadu_ps(other->data + i);

        sum = _mm_add_ps(a,b);

        _mm_storeu_ps(temp,sum);

        for (int j = 0; j < 4; j++)
        {
            (*result)[i + j] = temp[j];
        }
    }
    for (; i < size; i++)
    {
        (*result)[i] = (*this)[i] + (*other)[i];
    }


    delete[] temp;
    
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

void Matrix::MultiplyAllDims(float value)
{
    int size = this->rows * this->cols * this->dim;

    for (int i = 0; i < size; i++)
    {
        this->data[i] *= value;
    }
}


void Matrix::DivideAllDims(float value)
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
            if (j != matrix.cols - 1)
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
    std::cout << "(" << this->rows << "," << this->cols << "," << this->dim << ")" << std::endl;
}

float& Matrix::operator[](int index)
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

        
    return data[index];

}

const float& Matrix::operator[](int index) const
{

#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return this->data[index];
}

const float& Matrix::operator()(int _rows, int _cols) const
{
#if SAFE
    if (_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    return data[_rows * this->cols + _cols];
}

const float& Matrix::operator()(int _rows, int _cols, int _dims) const
{
#if SAFE
    if (_rows >= rows || _cols >= cols || _dims >= dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif
    return data[_dims * matrixSize + _rows * cols + _cols];
}

Matrix* Matrix::operator*=(const Matrix* other)
{
#if SAFE
    if (this->cols != other->cols && this->rows != other->rows)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
#endif

    float* temp = new float[4];

    int size = this->rows * this->cols;
    int i;
    for (i = 0; i + 4 <= size; i+=4)
    {
        __m128 sum = _mm_setzero_ps();
        __m128 a = _mm_loadu_ps(data + i);
        __m128 b = _mm_loadu_ps(other->data + i);

        sum = _mm_mul_ps(a,b);


        _mm_storeu_ps(temp,sum);

        for (int j = 0; j < 4; j++)
        {
            (*this)[i + j] = temp[j];
        }
    }
    for (; i < size; i++)
    {
        (*this)[i] *= (*other)[i];
    }


    delete[] temp;

    return this;

/*
    for (int i = 0; i < cols * rows; i++)
    {
        this->data[i] *= other->data[i];
    }
    return this;

*/
}

Matrix* Matrix::operator*=(const float other)
{
    for (int i = 0; i < cols * rows; i++)
    {
        this->data[i] = data[i] * other;
    }
    return this;
}


Matrix* Matrix::operator+(const Matrix& other) const
{
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    auto* result = new Matrix(this->rows, this->cols);
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


Matrix* Matrix::operator*(const float& other)
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
    if (this->rows != other.rows || this->cols != other.cols)
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

void Matrix::CrossProduct(const Matrix* other, Matrix* output) const
{
#if SAFE

    if (other->rows != this->cols)
    {
        throw std::runtime_error("Matrice have not the shape to be cross producted !");
    }
    if (output->rows != this->rows || output->cols != other->cols)
    {
        throw std::runtime_error("Output matrix has not the right shape !");
    }

#endif

/*
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

*/
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < other->cols; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= this->cols - 4; k += 4)
            {
                __m128 a = _mm_loadu_ps(&this->data[i * this->cols + k]);
                __m128 b = _mm_loadu_ps(&other->data[k * other->cols + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            
            float temp[4];
            _mm_storeu_ps(temp, sum);
            output->data[i * output->cols + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < this->cols; ++k)
            {
                output->data[i * output->cols + j] += this->data[i * this->cols + k] * other->data[k * other->cols + j];
            }
        }
    }



}


//Read and write matrices.

Matrix* Matrix::Read(std::ifstream& reader)
{
    int row, col, dim;
    reader.read(reinterpret_cast<char*>(&row), sizeof(int));
    reader.read(reinterpret_cast<char*>(&col), sizeof(int));
    reader.read(reinterpret_cast<char*>(&dim), sizeof(int));
    auto* matrix = new Matrix(row, col, dim);
    for (int i = 0; i < row * col * dim; i++)
    {
        reader.read(reinterpret_cast<char*>(matrix->data + i), sizeof(float));
    }
    return matrix;
}

void Matrix::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&rows), sizeof(int));
    writer.write(reinterpret_cast<char*>(&cols), sizeof(int));
    writer.write(reinterpret_cast<char*>(&dim), sizeof(int));
    for (int i = 0; i < rows * cols * dim; i++)
    {
        writer.write(reinterpret_cast<char*>(data + i), sizeof(float));
    }
}

float Matrix::Distance(Matrix* a, Matrix* b)
{
#if SAFE
    if (a->cols != b->cols || a->rows != b->rows)
    {
        throw std::invalid_argument("Matrices need to have same size to calculate distance !");
    }
#endif
    float res = 0;
    for (int i = 0; i < a->cols * a->rows; i++)
    {
        res += (a[0][i] - b[0][i]) * (a[0][i] - b[0][i]);
    }
    res = std::sqrt(res);
    return res;
}

Matrix* Matrix::Copy()
{
    auto* resArray = new float[cols * rows * dim];
    for (int i = 0; i < cols * rows * dim; i++)
    {
        resArray[i] = data[i];
    }
    return new Matrix(rows, cols, dim, resArray);
}


Matrix* Matrix::CopyWithSameData()
{
    return new CloneMatrix(rows, cols, dim, data);
}

void Matrix::Flip180(const Matrix* input, Matrix* output)
{
    for (int i = 0; i < input->cols / 2; ++i)
    {
        for (int j = 0; j < input->rows / 2; ++j)
        {
            //UGLY
            (*output)(i, j) = (*input)(input->rows - 1 - j, input->cols - 1 - i);
        }
    }
}


void Matrix::GoToNextMatrix() const
{
    data += matrixSize;
    offset += matrixSize;
}

void Matrix::ResetOffset() const
{
    data -= offset;
    offset = 0;
}

void Matrix::SetOffset(const int offset_) const
{
    data += offset_;
    this->offset += offset_;
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
    if (output->cols != outputCols || outputRows != output->rows)
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
            float sum = 0;
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
                float sum = 0;
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
    if (outputCols != output->cols || output->rows != outputRows)
    {
        std::cout << outputRows << "\n";
        throw std::invalid_argument("Convolution : output matrix has not the right shape !");
    }
#endif

    for (int i = 0; i < output->rows; i++)
    {
        for (int j = 0; j < output->cols; j++)
        {
            float sum = 0;
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

float Matrix::Sum()
{
    float res = 0;
    for (int i = 0; i < cols * rows; i++)
    {
        res += data[i];
    }
    return res;
}

float& Matrix::operator()(int _rows, int _cols)
{
    if (_rows >= rows || _cols >= cols)
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

void Matrix::Reshape(const int rows_, const int cols_, const int dims) const
{
#if SAFE
    if (rows_ * cols_ * dims != this->cols * this->rows * this->dim)
    {
        throw std::invalid_argument("Reshape : Incorrect Reshape !");
    }
#endif

    this->rows = rows_;
    this->cols = cols_;
    this->dim = dims;
    matrixSize = rows_ * cols_;
}

int Matrix::size() const
{
    return matrixSize * dim;
}


Matrix* Matrix::Copy(const Matrix* a)
{
    auto* res = new Matrix(a->rows, a->cols, a->dim);
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
                float max = -DBL_MAX;
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


Matrix Matrix::Random(const int rows, const int cols)
{
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        res[i] = (float) rand() / RAND_MAX * 2 - 1;

    return res;
}

float* Matrix::GetData()
{
    return data;
}

Matrix* Matrix::operator/=(const float other)
{
    for (int i = 0; i < rows * cols * dim; i++)
    {
        data[i] /= other;
    }
    return this;
}


//MATRIX CARRE

MatrixCarre::MatrixCarre(int size) : Matrix(size, size)
{
    this->operator[](0) = 1;
}

MatrixCarre::MatrixCarre(int size, float value) : Matrix(size, size)
{
    for (int i = 0; i < size * size; i++)
    {
        this->operator[](i) = value;
    }
}

//MATRIX DIAGONALE

MatrixDiagonale::MatrixDiagonale(int size, float value) : Matrix(size, size)
{
    for (int i = 0; i < size; i++)
    {
        this->operator[](i * size + i) = value;
    }
}

MatrixDiagonale::~MatrixDiagonale()
{

}




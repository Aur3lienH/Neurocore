#include <cmath>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include "Matrix.cuh"

#if USE_GPU

#include <cublas.h>

#endif

#define SAFE false

#if USE_GPU

Matrix_GPU::Matrix_GPU(const int rows, const int cols, const int dims) : rows(rows), cols(cols), dims(dims),
                                                                         matrixSize(rows * cols),
                                                                         size(rows * cols * dims)
{
    checkCUDA(cudaMalloc(&data_d, rows * cols * dims * sizeof(float)));
    checkCUDA(cudaMemset(data_d, 0, rows * cols * dims * sizeof(float)));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
}

Matrix_GPU::Matrix_GPU(const Matrix& mat) : Matrix_GPU(mat.GetRows(), mat.GetCols(), mat.GetDims())
{
    checkCUDA(cudaMemcpy(data_d, mat.GetData(), mat.GetSize() * sizeof(float), cudaMemcpyHostToDevice));
}

Matrix_GPU::~Matrix_GPU()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(desc));
    cudaFree(data_d);
}

void Matrix_GPU::Zero()
{
    cudaMemset(data_d, 0, size * sizeof(float));
}

float* Matrix_GPU::GetData_CPU() const
{
    float* data_h;
    cudaMemcpy(data_h, data_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    return data_h;
}

void Matrix_GPU::DivideAllDims(const float factor)
{
    const float multFactor = 1.f / factor;
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc, data_d, &multFactor));
}

void Matrix_GPU::Multiply(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res)
{
    throw std::runtime_error("Matrix_GPU::Multiply not implemented yet !");
    // Help to deal with CUBLAS fuckin column-major order
    //https://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
    /*checkCUBLAS(cublasSgeam(cublasHandle,
                               cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n,
                               const float* alpha,
                               const float* A, int lda,
                               const float* beta,
                               const float* B, int ldb,
                               float* C, int ldc));*/
}

void Matrix_GPU::Add(const Matrix_GPU& other, Matrix_GPU& res)
{
    throw std::runtime_error("How to handle difference btw Add and AddAllDims ? (because of tensor descriptors)");
    checkCUDNN(cudnnAddTensor(cuda->cudnnHandle, &one, desc, data_d, other.data_d, desc, data_d));
}

Matrix_GPU* Matrix_GPU::operator*=(const float n)
{
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc, data_d, &n));
}

void Matrix_GPU::Reshape(int rows_, int cols_, int dims_) const
{
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1, size, 1));
    rows = rows_;
    cols = cols_;
    dims = dims;
}

void Matrix_GPU::Flatten() const
{
    Reshape(rows * cols * dims, 1, 1);
}

float Matrix_GPU::GetAt(int index) const
{
    float res;
    checkCUDA(cudaMemcpy(&res, data_d + index, sizeof(float), cudaMemcpyDeviceToHost));

    return res;
}

void Matrix_GPU::SetAt(const int index, float value)
{
    checkCUDA(cudaMemcpy(data_d + index, &value, sizeof(float), cudaMemcpyHostToDevice));
}

int Matrix_GPU::GetRows() const
{
    return rows;
}

int Matrix_GPU::GetCols() const
{
    return cols;
}

int Matrix_GPU::GetDims() const
{
    return dims;
}

int Matrix_GPU::GetSize() const
{
    return size;
}

float* Matrix_GPU::GetData() const
{
    return data_d;
}

cudnnTensorDescriptor_t* Matrix_GPU::GetDescriptor() const
{
    return &desc;
}

Matrix_GPU* Matrix_GPU::Copy() const
{
    auto* res = new Matrix_GPU(rows, cols, dims);
    checkCUDA(cudaMemcpy(res->data_d, data_d, size * sizeof(float), cudaMemcpyDeviceToDevice));
    return res;
}

void Matrix_GPU::Save(std::ofstream& writer) const
{
    std::cout << "I was lazy to implement this function, sorry !\n";
    Matrix cpy(rows, cols, GetData_CPU());
    cpy.Save(writer);
}

Matrix_GPU* Matrix_GPU::CopyWithSameData() const
{
    CloneMatrix_GPU* res = new CloneMatrix_GPU(rows, cols, dims);
    res->data_d = data_d;

    return res;
}

#endif

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
    if (aligned)
    {
        if (posix_memalign((void**) &data, 32, sizeof(float) * rows * cols * dim))
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

int Matrix::GetRows() const
{
    return this->rows;
}

int Matrix::GetCols() const
{
    return this->cols;
}

int Matrix::GetDims() const
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
    for (i = 0; i + 4 <= size; i += 4)
    {
        __m128 sum = _mm_setzero_ps();
        __m128 a = _mm_loadu_ps(data + i);
        __m128 b = _mm_loadu_ps(other->data + i);

        sum = _mm_add_ps(a, b);

        _mm_storeu_ps(temp, sum);

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
    for (i = 0; i + 4 <= size; i += 4)
    {
        __m128 sum = _mm_setzero_ps();
        __m128 a = _mm_loadu_ps(data + i);
        __m128 b = _mm_loadu_ps(other->data + i);

        sum = _mm_mul_ps(a, b);


        _mm_storeu_ps(temp, sum);

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
    const int outputCols = m->GetCols() + filter->GetCols() - 1;
    const int outputRows = m->GetRows() + filter->GetRows() - 1;

#if SAFE
    if (output->cols != outputCols || outputRows != output->rows)
    {
        std::cout << "right shape is : " << "(" << outputRows << "," << outputCols << ")\n";
        throw std::invalid_argument("FullConvolution : Output Matrix has not the right shape ! ");
    }
#endif
    const int filterCols = filter->GetCols();
    const int filterRows = filter->GetRows();

    const int inputCols = m->GetCols();
    const int inputRows = m->GetRows();

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

/*
void Matrix::FullConvolutionAVX2(const Matrix* m, const Matrix* filter, Matrix* output)
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

    const int inputCols = m->GetCols();
    const int inputRows = m->GetRows();

    const int r = filterRows - 1;
    const int c = filterCols - 1;
    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            float sum = 0;
            __m256 v_sum = _mm256_setzero_ps();
            
            for (int k = 0; k < filterRows; k++)
            {
                int l = 0;
                for (; l + 7 < filterCols; l += 8) // Process in chunks of 8 where possible
                {
                    const int inputRow = i + k - r;
                    const int inputCol = j + l - c;
                    
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol + 7 < inputCols)
                    {
                        __m256 v_m = _mm256_loadu_ps(&(*m)(inputRow, inputCol));
                        __m256 v_filter = _mm256_loadu_ps(&(*filter)(k, l));
                        __m256 v_product = _mm256_mul_ps(v_m, v_filter);
                        
                        v_sum = _mm256_add_ps(v_sum, v_product);
                    }
                }

                // Cleanup loop for any remaining elements
                for (; l < filterCols; l++)
                {
                    const int inputRow = i + k - r;
                    const int inputCol = j + l - c;

                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                    {
                        sum += (*m)(inputRow, inputCol) * (*filter)(k, l);
                    }
                }
            }

            // Horizontally add the results in v_sum
            float arr[8];
            _mm256_storeu_ps(arr, v_sum);
            for(int p = 0; p < 8; p++) sum += arr[p];

            (*output)(i, j) += sum;
        }
    }

}

*/

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
    int inputCols = input->GetCols();
    int inputRows = input->GetRows();
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
            for (int k = 0; k < filter->GetRows(); k++)
            {
                for (int l = 0; l < filter->GetRows(); l++)
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

int Matrix::GetSize() const
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

float* Matrix::GetData() const
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

Matrix* Matrix::Transpose() const
{
    auto* res = new Matrix(cols, rows, dim);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols * dim; j++)
        {
            res->data[j * rows + i] = data[i * cols * dim + j];
        }
    }
    return res;
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




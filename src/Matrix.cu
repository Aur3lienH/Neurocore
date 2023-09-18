#include <cmath>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <emmintrin.h>
#include <cstdlib>
#include "Matrix.cuh"

#define SAFE 0
#define AVX2 false
#define SSE2 false



//MATRIX

Matrix::Matrix()
{

}

void Matrix::Init(const int rows, const int cols, const int dim, const float value, bool aligned)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dim;
    matrixSize = rows * cols;
    if (aligned)
    {
        //Create a matrix aligned by 32
        if (posix_memalign((void**) &data, 32, sizeof(float) * rows * cols * dim))
        {
            throw std::invalid_argument("Cannot create an aligned array ! ");
        }
    }
    else
    {
        //Create a simple array of size rows * cols * dim
        this->data = new float[rows * cols * dim];
    }


    //Make all the values = 0
    for (int i = 0; i < rows * cols * dim; i++)
    {
        data[i] = 0;
    }
}


Matrix::Matrix(const int rows, const int cols, bool aligned)
{
    Init(rows, cols, 1, aligned);
}


Matrix::Matrix(const int rows, int cols, int dim, bool aligned)
{
    Init(rows, cols, dim, 0, aligned);
}

//Initialize a matrix with a default value
Matrix::Matrix(const int rows, const int cols, float value, bool aligned)
{
    Init(rows, cols, 1, value, aligned);
}


//Initialize a matrix with an array already existing
Matrix::Matrix(const int rows, const int cols, float* newArray)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = 1;
    this->data = newArray;
    matrixSize = rows * cols;
}

//Initialize a 3D matrix with an already existing array
Matrix::Matrix(const int rows, const int cols, const int dims, float* data)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dims;
    this->data = data;
    matrixSize = rows * cols;
}

#if USE_GPU

Matrix_GPU::Matrix_GPU(const int rows, const int cols, const int dims) : rows(rows), cols(cols), dims(dims),
                                                                         matrixSize(rows * cols),
                                                                         size(rows * cols * dims), offset(0)
{
    checkCUDA(cudaMalloc(&data_d, rows * cols * dims * sizeof(float)));
    checkCUDA(cudaMemset(data_d, 0, rows * cols * dims * sizeof(float)));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_1D));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
}

Matrix_GPU::Matrix_GPU(const Matrix& mat) : Matrix_GPU(mat.GetRows(), mat.GetCols(), mat.GetDims())
{
#if SAFE
    if (size != mat.GetSize())
        throw std::runtime_error("Matrices dimensions do not match !");
#endif
    checkCUDA(cudaMemcpy(data_d, mat.GetData(), mat.GetSize() * sizeof(float), cudaMemcpyHostToDevice));
}

Matrix_GPU::~Matrix_GPU()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(desc_1D));
    cudaFree(data_d);
}

void Matrix_GPU::Zero()
{
    checkCUDA(cudaMemset(data_d, 0, size * sizeof(float)));
}

float* Matrix_GPU::GetData_CPU() const
{
    float* data_h = new float[size];
    checkCUDA(cudaMemcpy(data_h, data_d - offset, size * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

float* Matrix_GPU::GetData_CPU_1D() const
{
    float* data_h = new float[matrixSize];
    checkCUDA(cudaMemcpy(data_h, data_d, matrixSize * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

void Matrix_GPU::DivideAllDims(const float factor)
{
    const float multFactor = 1.f / factor;
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc, data_d, &multFactor));
}

void Matrix_GPU::Multiply(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res)
{
    // Help to deal with CUBLAS fucking column-major order
    //https://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
    //https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp#L293
    checkCUBLAS(cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, b.GetCols(), a.GetRows(), a.GetCols(),
                               &cuda->one, b.data_d, b.GetCols(), a.data_d, a.GetCols(), &cuda->zero, res.data_d,
                               b.GetCols()));
}

void Matrix_GPU::MultiplyByTransposeAndAddToRes(const Matrix_GPU& other, Matrix_GPU& res)
{
    checkCUBLAS(
            cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, other.rows, rows, cols, &cuda->one,
                           other.data_d, other.cols, data_d, cols, &cuda->one, res.data_d, other.rows));
}

void Matrix_GPU::MultiplyTransposeBy(const Matrix_GPU& other, Matrix_GPU& res)
{
    /*int values[] = {rows, cols, other.rows, other.cols};
    for (const int m : values)
    {
        for (const int n : values)
        {
            for (const int k : values)
            {
                for (const int lda : values)
                {
                    for (const int ldb : values)
                    {
                        for (const int ldc : values)
                        {
                            std::cout << "M: " << m << ", N: " << n << ", K: " << k << ", ldA: " << lda << ", ldB: "
                                      << ldb << ", ldC: " << ldc << std::endl;
                            try
                            {
                                checkCUBLAS(
                                        cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                                       &cuda->one,
                                                       other.data_d, lda, data_d, ldb, &cuda->zero, res.data_d,
                                                       ldc));
                                std::cout << "Success !\n";
                                if (std::abs(res.GetAt(0) - 0.0109781f) < .01)
                                    std::cout << res << std::endl << std::flush;
                            }
                            catch (std::exception& e)
                            {
                                std::cout << "Fail" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }*/
    checkCUBLAS(
            cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, other.cols, cols, other.rows, &cuda->one,
                           other.data_d, other.cols, data_d, cols, &cuda->zero, res.data_d, other.cols));
}

void Matrix_GPU::Add(const Matrix_GPU& other, Matrix_GPU& res)
{
#if SAFE
    if (matrixSize != other.matrixSize)
        throw std::runtime_error("Matrices dimensions mismatch");
#endif
    checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                        &cuda->one, data_d, rows, &cuda->one, other.data_d, rows, res.data_d, rows));
}

Matrix_GPU* Matrix_GPU::operator*=(const float n)
{
    //checkCUBLAS(cublasSscal_v2(cuda->cublasHandle, matrixSize, &n, data_d, 1));
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc_1D, data_d, &n));

    return this;
}

void Matrix_GPU::Reshape(int rows_, int cols_, int dims_) const
{
    rows = rows_;
    cols = cols_;
    dims = dims_;
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
}

void Matrix_GPU::Flatten() const
{
    Reshape(size, 1, 1);
}

float Matrix_GPU::GetAt(const int index) const
{
    float res;
    checkCUDA(cudaMemcpy(&res, data_d + index, sizeof(float), cudaMemcpyDeviceToHost));

    return res;
}

void Matrix_GPU::SetAt(const int index, const float value)
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

int Matrix_GPU::GetMatrixSize() const
{
    return matrixSize;
}

Matrix_GPU* Matrix_GPU::Copy() const
{
    auto* res = new Matrix_GPU(rows, cols, dims);
    checkCUDA(cudaMemcpy(res->data_d, data_d, size * sizeof(float), cudaMemcpyDeviceToDevice));
    return res;
}

void Matrix_GPU::Save(std::ofstream& writer) const
{
    Matrix cpy(rows, cols, GetData_CPU());
    cpy.Save(writer);
}

Matrix_GPU* Matrix_GPU::CopyWithSameData() const
{
    CloneMatrix_GPU* res = new CloneMatrix_GPU(rows, cols, dims);
    res->data_d = data_d;

    return res;
}

__global__
void CoeffWiseMultKernel(const float* a, const float* b, float* res, const int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        res[index] = a[index] * b[index];

}

void Matrix_GPU::HadamardProduct(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res)
{
    std::cout << "This is done on all Dims\n";
#if SAFE
    if (a.GetSize() != b.GetSize())
        throw std::runtime_error("Matrices dimensions mismatch !");
#endif
    const int size = a.GetSize();
    const int blocksPerGrid = (size + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock;
    CoeffWiseMultKernel<<<blocksPerGrid, cuda->threadsPerBlock>>>(a.data_d, b.data_d, res.data_d, size);
    checkCUDA(cudaDeviceSynchronize());
}

std::ostream& operator<<(std::ostream& os, const Matrix_GPU& matrix)
{
    std::cout << "Matrix: " << matrix.rows << "x" << matrix.cols << std::endl;
    float* data = matrix.GetData_CPU() + matrix.offset;
    for (int i = 0; i < matrix.rows; i++)
    {
        os << "[";
        for (int j = 0; j < matrix.cols; j++)
        {
            os << data[i * matrix.cols + j];
            if (j != matrix.cols - 1)
            {
                os << " ";
            }
        }
        os << "]\n";
    }

    data -= matrix.offset;
    delete data;

    return os;
}

void Matrix_GPU::DisplayTensorInfo(cudnnTensorDescriptor_t const& desc)
{
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    std::cout << "DataType: " << dataType << ", n: " << n << ", c: " << c << ", h: " << h << ", w: " << w
              << ", nStride: "
              << nStride << ", cStride: " << cStride << ", hStride: " << hStride << ", wStride: " << wStride
              << std::endl;
}

#endif

//Deallocating the matrix
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

bool Matrix::IsColumnMajor() const
{
    return columnMajor;
}


//Add two matrix using SSE2 SMID instructions
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

void Matrix::Print() const 
{
    Matrix matrix = *this;
    std::cout << "Matrix: " << matrix.rows << "x" << matrix.cols << std::endl;


    if(matrix.columnMajor)
    {
        for (int i = 0; i < matrix.cols; i++)
        {
            std::cout << "[";
            for (int j = 0; j < matrix.rows; j++)
            {
                std::cout << matrix.data[i + j * matrix.rows];
                if (j != matrix.cols - 1)
                {
                    std::cout << " ";
                }
            }
            std::cout << "]\n";
        }
    }
    else
    {
        for (int i = 0; i < matrix.rows; i++)
        {
            std::cout << "[";
            for (int j = 0; j < matrix.cols; j++)
            {
                std::cout << matrix.data[i * matrix.cols + j];
                if (j != matrix.cols - 1)
                {
                    std::cout << " ";
                }
            }
            std::cout << "]\n";
        }
    }
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

void Matrix::MatrixMultiplication(const Matrix* other, Matrix* output) const
{
#if SAFE

    if (other->rows != this->cols)
    {
        throw std::runtime_error("Matrix have not the shape to be cross producted !");
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


void Matrix::CrossProductWithSelfTranspose(const Matrix* other, Matrix* output) const
{
#if SAFE

    if (other->rows != this->rows)
    {
        throw std::runtime_error("Matrix have not the shape to be cross producted !");
    }
    if (output->rows != this->cols || output->cols != other->cols)
    {
        throw std::runtime_error("Output matrix has not the right shape !");
    }
#endif

    /*for (int i = 0; i < this->cols; i++)
    {
        for (int j = 0; j < other->cols; j++)
        {
            output->data[i * other->cols + j] = 0;
            for (int k = 0; k < this->rows; k++)
            {
                output->data[i * output->cols + j] += this->data[k * this->cols + i] * other->data[k * other->cols + j];
            }
        }
    }*/

    //sse2 version
    for (int i = 0; i < this->cols; i++)
    {
        for (int j = 0; j < other->cols; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= this->rows - 4; k += 4)
            {
                __m128 a = _mm_loadu_ps(&this->data[k * this->cols + i]);
                __m128 b = _mm_loadu_ps(&other->data[k * other->cols + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }

            float temp[4];
            _mm_storeu_ps(temp, sum);
            output->data[i * output->cols + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if rows is not a multiple of 4
            for (; k < this->rows; ++k)
            {
                output->data[i * output->cols + j] += this->data[k * this->cols + i] * other->data[k * other->cols + j];
            }
        }
    }
}

void Matrix::CrossProductWithTranspose(const Matrix* other, Matrix* output) const
{
    /*for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < other->rows; j++)
        {
            output->data[i * other->rows + j] = 0;
            for (int k = 0; k < this->cols; k++)
            {
                output->data[i * output->cols + j] += this->data[i * this->cols + k] * other->data[j * other->cols + k];
            }
        }
    }*/

    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < other->rows; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= this->cols - 4; k += 4)
            {
                __m128 a = _mm_loadu_ps(&this->data[i * this->cols + k]);
                __m128 b = _mm_loadu_ps(&other->data[j * other->cols + k]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }

            float temp[4];
            _mm_storeu_ps(temp, sum);
            output->data[i * output->cols + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < this->cols; ++k)
            {
                output->data[i * output->cols + j] += this->data[i * this->cols + k] * other->data[j * other->cols + k];
            }
        }
    }
}




void Matrix::OptimizedCrossProduct(const Matrix* a, const Matrix* other, Matrix* output)
{

#if SAFE

    if (a->rows != a->cols)
    {
        throw std::runtime_error("Matrice have not the shape to be cross producted !");
    }
    if (a->rows != a->rows || output->cols != other->cols)
    {
        throw std::runtime_error("Output matrix has not the right shape !");
    }

#endif


    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < other->cols; j++)
        {
            output->data[i * other->cols + j] = 0;
            int k = 0;
#if AVX2
            __m256 sum256 = _mm256_setzero_ps();
            for (; k <= a->cols - 8; k += 8)
            {
                __m256 m_a = _mm256_load_ps(&a->data[i * a->cols + k]);


                __m256 b = _mm256_loadu_ps(&a->data[k * other->cols + j]);
                sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(m_a, b));
            }
            float temp256[8];
            //sum256 = _mm256_hadd_ps(sum256,sum256);
            //sum256 = _mm256_hadd_ps(sum256,sum256);
            _mm256_storeu_ps(temp256,sum256);
            output->data[i * output->cols + j] += temp256[0] + temp256[1] + temp256[2] + temp256[3] + temp256[4] + temp256[5] + temp256[6] + temp256[7];
#endif


#if SSE2
            __m128 sum = _mm_setzero_ps();
            for (; k <= a->cols - 4; k += 4)
            {
                __m128 m_a = _mm_loadu_ps(&a->data[i * a->cols + k]);
                __m128 b = _mm_loadu_ps(&other->data[k * other->cols + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(m_a, b));
            }

            float temp[4];
            sum = _mm_hadd_ps(sum,sum);
            sum = _mm_hadd_ps(sum,sum);
            _mm_storeu_ps(temp, sum);
            output->data[i * output->cols + j] += temp[0];
#endif



            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < a->cols; ++k)
            {
                output->data[i * output->cols + j] += a->data[i * a->cols + k] * other->data[k * other->rows + j];
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


void Matrix::FullConvolutionFS4(const Matrix* m, const Matrix* filter, Matrix* output)
{

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
    int filterSize = filter->GetRows();
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

bool Matrix::operator==(const Matrix other)
{
    if (other.GetRows() != this->GetRows() || other.cols != this->GetCols() || other.dim != this->GetDims())
    {
        return false;
    }

    for (int i = 0; i < this->GetSize(); i++)
    {
        if (abs(other.data[i] - this->data[i]) > 0.0001f)
        {
            return false;
        }
    }

    return true;

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

bool Matrix::IsNull(const Matrix* matrix)
{
    bool isNull = true;
    for (int i = 0; i < matrix->GetSize(); i++)
    {
        if ((*matrix)[i] != 0)
        {
            isNull = false;
            break;
        }
    }
    return isNull;

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




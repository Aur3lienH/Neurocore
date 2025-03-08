#pragma once
#include <cfloat>
#include <iostream>
#include <vector>
#include <cmath>
#include <emmintrin.h>
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cudnn.h"
#include "gpuComputation/CUDA.cuh"

#define SAFE 0
#define AVX2 false
#define SSE2 false

namespace py = pybind11;

template<int rows = 1, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class Matrix final {
public:
    Matrix(std::initializer_list<float> values) {
        Init(0);

        if constexpr (GPU) {
            float *vals = new float[values.size()];
            int i = 0;
            for (const float &value: values)
                vals[i++] = value;
            checkCUDA(
                cudaMemcpy(data_d, vals, values.size() * sizeof(float), cudaMemcpyHostToDevice
                ));
            delete[] vals;
        } else {
            int i = 0;
            for (const float &value: values) {
                if (i < GetMatrixSize()) {
                    data[i++] = value;
                }
            }
        }
    }

    Matrix(py::array_t<float> &input) {
        py::buffer_info buf = input.request();

        float *ptr = static_cast<float *>(buf.ptr);

        Init(0);
        if constexpr (GPU) {
            checkCUDA(cudaMemcpy(data_d, ptr, GetMatrixSize() * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            memcpy(this->data, ptr, GetMatrixSize() * sizeof(float));
        }
    }

    static py::object ConvertToArray(py::array_t<float> &input) {
        py::buffer_info buf = input.request();
        auto shape = buf.shape;
        float *ptr = static_cast<float *>(buf.ptr);
        size_t n_matrices = 1;
        size_t matrix_elements;

        // Handle different dimensions
        if (shape.size() == 3) {
            // Format: [n_matrices, rows, cols]
            n_matrices = shape[0];
            matrix_elements = shape[1] * shape[2];
        } else if (shape.size() == 2) {
            // Format: [n_matrices, matrix_elements] or [rows, cols]
            if (shape[1] == rows * cols) {
                // First dim is number of matrices
                n_matrices = shape[0];
                matrix_elements = shape[1];
            } else {
                // Single matrix case
                matrix_elements = shape[0] * shape[1];
            }
        } else {
            // Handle error or single dimension case
            matrix_elements = shape[0];
        }

        // Create and populate matrices
        Matrix *matrices = new Matrix[n_matrices];
        for (size_t i = 0; i < n_matrices; i++) {
            matrices[i].Init(0);
            if constexpr (GPU)
            {
                cudaMemcpy(matrices[i].data_d, ptr, matrices[i].GetMatrixSize() * sizeof(float), cudaMemcpyHostToDevice);
            }
            else
            {
                memcpy(matrices[i].data, ptr + (i * matrix_elements), matrix_elements * sizeof(float));
            }
        }

        return py::capsule(matrices, "MATRIX_ARRAY", [](void *f) {
            delete[] reinterpret_cast<Matrix *>(f);
        });
    }


    py::array_t<float> ToNumpy() {
        std::vector<size_t> shape;

        // Ne mettre dans shape que les dimensions > 1
        if (rows > 1) shape.push_back(rows);
        if (cols > 1) shape.push_back(cols);
        if (dims > 1) shape.push_back(dims);
        owner = false;
        // Si toutes les dimensions sont 1, on retourne un array 1D avec un seul élément
        //owner = false;
        if (shape.empty()) {
            shape.push_back(1);
        }

        return py::array_t<float>(shape, data);
    }


    explicit Matrix();

    explicit Matrix(float value);

    explicit Matrix(float *newArray, bool owner = false);

    ~Matrix();

    void GPU_init() requires(GPU);

public:
    static void Flip180(const Matrix *input, Matrix *output);

    template<int filter_rows, int filter_cols>
    static void FullConvolution(const Matrix<rows, cols> *m, const Matrix<filter_rows, filter_cols> *filter,
                                Matrix<rows + filter_rows - 1, cols + filter_cols - 1, dims> *output);

    //static void FullConvolutionAVX2(const Matrix* m, const Matrix* filter, Matrix* output);

    //FullConvolution FS4 = Filter Size 4
    //static void FullConvolutionFS4(const Matrix* m, const Matrix* filter, Matrix* output);

    template<int filterSize, int stride>
    static void Convolution(const Matrix<rows, cols, dims> *input, const Matrix<filterSize, filterSize, dims> *filter,
                            Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1, dims> *output);


    template<int filterSize, int stride>
    static void MaxPool(const Matrix<rows, cols, dims, GPU> *a,
                        Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1> *output);

    template<int filterSize, int stride>
    static void AveragePool(const Matrix<rows, cols, dims, GPU> *a,
                            Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1> *output);

    static Matrix Random();

    Matrix<cols, rows, dims, GPU> *Transpose() const;


    //Movement threw the matrix with the offset, all the operations are done with matrix with this offset
    void GoToNextMatrix() const;

    void ResetOffset() const;

    void SetOffset(int offset_) const;

    int GetOffset() const;

    float *GetData() const;

    //  In a template Pattern, cannot happend !
    //    void Flatten() const;

    //Cannot happen either !
    //void Reshape(int rows_, int cols_, int dimss) const;

    void Add(Matrix *other, Matrix *result);

    void AddAllDims(Matrix *other, Matrix *result);

    void Substract(const Matrix *other, Matrix *result) const;

    void SubstractAllDims(const Matrix *other, Matrix *result) const;

    void MultiplyAllDims(const Matrix *other, Matrix *result) const;

    void MultiplyAllDims(float value);

    void DivideAllDims(float value);

    void Zero();

    float Sum();

    constexpr int GetRows() const;

    constexpr int GetCols() const;

    constexpr int GetDims() const;

    constexpr int GetSize() const;

    constexpr int GetMatrixSize() const;

    Matrix *operator+=(const Matrix &other);

    Matrix *operator-=(const Matrix &other);

    Matrix *operator+(const Matrix &other) const;

    Matrix *operator-(const Matrix &other) const;

    Matrix *operator*=(const Matrix *other) requires(!GPU);

    Matrix *operator*=(float other);

    Matrix *operator/=(float other);

    Matrix *operator*(const float &other);

    bool operator==(const Matrix other);

    static Matrix *Read(std::ifstream &reader);

    void Save(std::ofstream &writer);

    float get(int index) const;

    float get(int _rows, int _cols) const;

    void set(int index, float value);

    void set(int _rows, int _cols, float value);

    //float& operator[](int index) requires(!GPU);

    //float& operator()(int _rows, int _cols) requires(!GPU);  // () cannot be used to set values ton GPU matrices

    //const float& operator[](int index) const requires(!GPU);;

    //const float& operator()(int _rows, int _cols) const requires(!GPU);;

    //const float& operator()(int _rows, int _cols, int _dims) const requires(!GPU);;

    template<int other_rows, int other_cols>
    void MatrixMultiplication(const Matrix<other_rows, other_cols> *other, Matrix<rows, other_cols> *output) const;

    void CrossProductWithTranspose(const Matrix *other, Matrix *output) const;

    void CrossProductWithSelfTranspose(const Matrix *other, Matrix *output) const;

    static void OptimizedCrossProduct(const Matrix *a, const Matrix *other, Matrix *output);


    void Print() const;

    void PrintSize() const;

    static float Distance(Matrix *a, Matrix *b);

    Matrix *Copy();

    Matrix *CopyWithSameData();

    static Matrix *Copy(const Matrix *a);

    static bool IsNull(const Matrix *a);

    bool IsColumnMajor() const;

    float *GetData_CPU() const requires(GPU);

    float *GetData_CPU_1D() const requires(GPU);

    Matrix<rows, cols, dims, false> *CPU_copy() const requires(GPU);

    template<int other_rows>
    void MultiplyByTransposeAndAddToRes(const Matrix<other_rows, cols> &other, Matrix<rows, other_rows> &res) requires(
        GPU);

    template<int other_cols>
    void MultiplyTransposeBy(const Matrix<rows, other_cols> &other, Matrix<cols, other_cols> &res) requires(GPU);

    //std::vector<Operation*> O_CrossProduct(Matrix* a, Matrix* b, Matrix* output);

    mutable float *data = nullptr;
    cudnnTensorDescriptor_t desc;
    mutable cudnnTensorDescriptor_t desc_1D;
    float *data_d;

protected:
    mutable int offset = 0;
    bool columnMajor = false;
    bool owner = true;
    // Descriptor for the matrix to perform operations on a single dimension

private:
    void Init(float value = 0);
};

template<int x = 1, int y = 1, int z = 1, bool GPU = GPU_DEFAULT>
using MAT = Matrix<x, y, z, GPU>;

template<typename layershape>
using LMAT = MAT<layershape::x, layershape::y, layershape::z>;


//MATRIX
template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU>::Matrix() {
    Init(0);
}

template<int rows, int columns, int dims, bool GPU>
Matrix<rows, columns, dims, GPU>::Matrix(float value) {
    Init(value);
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Init(float value) {
    if constexpr (GPU) {
        checkCUDA(cudaMalloc(&data_d, rows * cols * dims * sizeof(float)));
        //checkCUDA(cudaMemset(data_d, 0, rows * cols * dims * sizeof(float)));
        checkCUDNN(cudnnCreateTensorDescriptor(&desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&desc_1D));
        checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
        checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
        checkKernel(initializeArray_kernel<<<CUDA_KERNEL_ARGS(cuda, GetSize())>>>(data_d, value, GetSize()));
    } else {
        //Create a simple array of size rows * cols * dim
        this->data = new float[rows * cols * dims];

        //Make all the values = value
        for (int i = 0; i < rows * cols * dims; i++) {
            data[i] = value;
        }
    }
}


//Initialize a matrix with an array already existing
template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU>::Matrix(float *newArray, bool _owner) {
    if constexpr (GPU) {
        checkCUDNN(cudnnCreateTensorDescriptor(&desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&desc_1D));
        checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
        checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
    }
    this->data = newArray;
    owner = _owner;
}

//Deallocating the matrix
template<int rows, int cols, int size, bool GPU>
Matrix<rows, cols, size, GPU>::~Matrix() {
    if constexpr (GPU) {
        checkCUDNN(cudnnDestroyTensorDescriptor(desc));
        checkCUDNN(cudnnDestroyTensorDescriptor(desc_1D));
    } else {
        cudaFree(data_d);
        if (owner) {
            delete[] this->data;
        }
    }
}

template<int rows, int cols, int size, bool GPU>
constexpr int Matrix<rows, cols, size, GPU>::GetRows() const {
    return rows;
}

template<int rows, int cols, int size, bool GPU>
constexpr int Matrix<rows, cols, size, GPU>::GetCols() const {
    return cols;
}

template<int rows, int cols, int size, bool GPU>
constexpr int Matrix<rows, cols, size, GPU>::GetDims() const {
    return size;
}

template<int rows, int cols, int dims, bool GPU>
constexpr int Matrix<rows, cols, dims, GPU>::GetMatrixSize() const {
    return rows * cols;
}


template<int rows, int cols, int size, bool GPU>
bool Matrix<rows, cols, size, GPU>::IsColumnMajor() const {
    return true;
}

//Add two matrix using SSE2 SMID instructions
template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Add(Matrix *other, Matrix *result) {
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        std::cout << "Matrix 1: " << this->rows << "x" << this->cols << std::endl;
        std::cout << "Matrix 2: " << other->rows << "x" << other->cols << std::endl;
        return;
    }
#endif
    if constexpr (GPU) {
        checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                &cuda->one, data_d, rows, &cuda->one, other->data_d, rows, result->data_d, rows));
    } else {
        /*
            for (int i = 0; i < this->rows * this->cols; i++)
            {
                result->data[i] = this->data[i] + other->data[i];
            }
        */

        float *temp = new float[4];

        int size = GetRows() * GetCols();
        int i;
        for (i = 0; i + 4 <= size; i += 4) {
            __m128 sum = _mm_setzero_ps();
            __m128 a = _mm_loadu_ps(data + i);
            __m128 b = _mm_loadu_ps(other->data + i);

            sum = _mm_add_ps(a, b);

            _mm_storeu_ps(temp, sum);

            for (int j = 0; j < 4; j++) {
                result->set(i + j, temp[j]);
            }
        }
        for (; i < size; i++) {
            result->set(i, get(i) + other->get(i));
        }


        delete[] temp;
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::AddAllDims(Matrix *other, Matrix *result) {
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    constexpr int size = this->GetRows() * this->GetCols() * this->GetDims();

    for (int i = 0; i < size; i++) {
        result->data[i] = this->data[i] + other->data[i];
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Substract(const Matrix *other, Matrix *result) const {
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif

    if constexpr (GPU) {
        checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                &cuda->one, data_d, rows, &cuda->minus_one, other->data_d, rows, result->data_d, rows));
    } else {
        for (int i = 0; i < this->GetRows() * this->GetCols(); i++) {
            result->data[i] = this->data[i] - other->data[i];
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
Matrix<cols, rows, dims, GPU> *Matrix<rows, cols, dims, GPU>::Transpose() const {
    auto *res = new Matrix<cols, rows, dims, GPU>();

    if constexpr (GPU) {
        for (int i = 0; i < dims; i++) {
            dim3 threads(16, 16); // Each block has 16x16 threads
            dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
            checkKernel((transpose_kernel<<<blocks, threads>>>(data_d, res->data_d, rows, cols)));
            //GoToNextMatrix();
            //res->GoToNextMatrix();
        }
        //ResetOffset();
        //res->ResetOffset();
    } else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols * dims; j++) {
                res->data[j * rows + i] = data[i * cols * dims + j];
            }
        }
    }
    return res;
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::SubstractAllDims(const Matrix *other, Matrix *result) const {
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    if constexpr (GPU)
    {
        for (auto i =0; i < GetDims(); i++)
        {
            Substract(other, result);
            GoToNextMatrix();
            other->GoToNextMatrix();
            result->GoToNextMatrix();
        }
        ResetOffset();
        other->ResetOffset();
        result->ResetOffset();
    }
    else
    {
        int size = GetRows() * GetCols() * GetDims();

        for (int i = 0; i < size; i++) {
            result->data[i] = this->data[i] - other->data[i];
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::MultiplyAllDims(const Matrix *other, Matrix *result) const {
#if SAFE
    if (this->rows != other->rows || this->cols != other->cols || this->dim != other->dim)
    {
        std::cout << "Error: Matrix dimensions must agree." << std::endl;
        return;
    }
#endif
    int size = GetRows() * GetCols() * GetDims();

    for (int i = 0; i < size; i++) {
        result->data[i] = this->data[i] * other->data[i];
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::MultiplyAllDims(float value) {
    if constexpr (GPU) {
        checkKernel((scalarMult_kernel<<<CUDA_KERNEL_ARGS(cuda, GetSize())>>>(data_d, value, data_d, GetSize())));
    } else {
        int size = GetRows() * GetCols() * GetDims();

        for (int i = 0; i < size; i++) {
            this->data[i] *= value;
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::DivideAllDims(float value) {
    if constexpr (GPU) {
        const float multFactor = 1.f / value;
        checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc, data_d, &multFactor));
    } else {
        int size = GetRows() * GetCols() * GetDims();

        for (int i = 0; i < size; i++) {
            this->data[i] /= value;
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Zero() {
    if constexpr (GPU) {
        checkCUDA(cudaMemset(data_d, 0, dims * sizeof(float)));
    } else {
        for (int i = 0; i < GetRows() * GetCols(); i++) {
            this->data[i] = 0;
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
float *Matrix<rows, cols, dims, GPU>::GetData_CPU() const requires(GPU) {
    const auto size = GetSize();
    float *data_h = new float[size];
    checkCUDA(cudaMemcpy(data_h, data_d - offset, size * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

template<int rows, int cols, int dims, bool GPU>
float *Matrix<rows, cols, dims, GPU>::GetData_CPU_1D() const requires(GPU) {
    const auto matrixSize = GetMatrixSize();
    float *data_h = new float[matrixSize];
    checkCUDA(cudaMemcpy(data_h, data_d, matrixSize * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, false> *Matrix<rows, cols, dims, GPU>::CPU_copy() const requires (GPU) {
    return new Matrix<rows, cols, dims, false>(GetData_CPU(), true);
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Print() const {
    std::cout << "Matrix: " << this->GetRows() << "x" << this->GetCols() << std::endl;

    float *d;
    if constexpr (GPU)
        d = GetData_CPU();
    else d = data;

    if (this->columnMajor) {
        for (int i = 0; i < this->GetCols(); i++) {
            std::cout << "[";
            for (int j = 0; j < this->GetRows(); j++) {
                std::cout << d[i + j * this->GetRows()];
                if (j != this->GetCols() - 1) {
                    std::cout << " ";
                }
            }
            std::cout << "]\n";
        }
    } else {
        for (int i = 0; i < this->GetRows(); i++) {
            std::cout << "[";
            for (int j = 0; j < this->GetCols(); j++) {
                std::cout << d[i * this->GetCols() + j];
                if (j != this->GetCols() - 1) {
                    std::cout << " ";
                }
            }
            std::cout << "]\n";
        }
    }

    if constexpr (GPU)
        delete[] d;
}


template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::PrintSize() const {
    std::cout << "(" << GetRows() << "," << GetCols() << "," << GetDims() << ")" << std::endl;
}

/*template <int rows, int cols, int dims, bool GPU>
float& Matrix<rows, cols, dims, GPU>::operator[](int index) requires(!GPU)
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return data[index];
}

template <int rows, int cols, int dims, bool GPU>
const float& Matrix<rows, cols, dims, GPU>::operator[](int index) const requires(!GPU)
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif
    if constexpr (GPU)
    {
        float val;
        checkCUDA(cudaMemcpy(&val, data_d + index, sizeof(float), cudaMemcpyDeviceToHost));
        return val;
    }

    return this->data[index];
}*/

/*template <int rows, int cols, int dims, bool GPU>
const float& Matrix<rows, cols, dims, GPU>::operator()(int _rows, int _cols) const
{
#if SAFE
    if (_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    return data[_rows * this->GetCols() + _cols];
}

template <int rows, int cols, int dims, bool GPU>
const float& Matrix<rows, cols, dims, GPU>::operator()(int _rows, int _cols, int _dims) const
{
#if SAFE
    if (_rows >= rows || _cols >= cols || _dims >= dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif
    return data[_dims * GetMatrixSize() + _rows * cols + _cols];
}*/

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator*=(const Matrix<rows, cols, dims, GPU> *other)
    requires(!GPU) {
#if SAFE
    if (this->cols != other->cols && this->rows != other->rows)
    {
        throw std::runtime_error("Error: Matrix dimensions must agree.");
    }
#endif


    float *temp = new float[4];

    int size = this->GetRows() * this->GetCols();
    int i;
    for (i = 0; i + 4 <= size; i += 4) {
        __m128 sum = _mm_setzero_ps();
        __m128 a = _mm_loadu_ps(data + i);
        __m128 b = _mm_loadu_ps(other->data + i);

        sum = _mm_mul_ps(a, b);


        _mm_storeu_ps(temp, sum);

        for (int j = 0; j < 4; j++) {
            data[i + j] = temp[j];
        }
    }
    for (; i < size; i++) {
        data[i] *= other->data[i];
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


template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator*=(const float other) {
    for (int i = 0; i < cols * rows; i++) {
        this->data[i] = data[i] * other;
    }
    return this;
}


template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator+(const Matrix &other) const {
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    auto *result = new Matrix();
    if constexpr (GPU) {
        checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                &cuda->one, data_d, rows, &cuda->one, other.data_d, rows, result->data_d, rows));
    } else {
        for (int i = 0; i < this->GetCols() * this->GetRows(); i++) {
            result->set(i, other.data[i] + this->data[i]);
        }
    }
    return result;
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator-(const Matrix &other) const {
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    auto *result = new Matrix();
    if constexpr (GPU) {
        checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                &cuda->one, data_d, rows, &cuda->minus_one, other.data_d, rows, result->data_d, rows));
    } else {
        for (int i = 0; i < this->GetCols() * this->GetRows(); i++) {
            result->set(i, this->data[i] - other.data[i]);
        }
    }
    return result;
}


template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator+=(const Matrix &other) {
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    for (int i = 0; i < this->GetCols() * this->GetRows(); i++) {
        this->data[i] += other.data[i];
    }
    return this;
}


template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator*(const float &other) {
    auto *result = new Matrix<rows, cols, dims, GPU>();

    if constexpr (GPU) {
        checkKernel(
            (scalarMult_kernel<<<CUDA_KERNEL_ARGS(cuda, GetMatrixSize())>>>(data_d, other, result->data_d, result->
                GetMatrixSize())));
    } else {
        for (int i = 0; i < this->GetCols() * this->GetRows(); i++) {
            result->data[i] = this->data[i] * other;
        }
    }
    return result;
}


template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator-=(const Matrix &other) {
#if SAFE
    if (this->rows != other.rows || this->cols != other.cols)
    {
        std::cout << "Matrices are not of the same size\n";
        return nullptr;
    }
#endif

    for (int i = 0; i < this->GetRows() * this->GetCols(); i++) {
        this->data[i] -= other.data[i];
    }
    return this;
}


template<int rows, int cols, int dims, bool GPU>
float Matrix<rows, cols, dims, GPU>::get(const int index) const {
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    if constexpr (GPU) {
        float val;
        checkCUDA(cudaMemcpy(&val, data_d + index, sizeof(float), cudaMemcpyDeviceToHost));
        return val;
    }

    return this->data[index];
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::set(const int index, const float value) {
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    if constexpr (GPU) {
        checkCUDA(cudaMemcpy(data_d + index, &value, sizeof(float), cudaMemcpyHostToDevice));
    } else { this->data[index] = value; }
}

template<int rows, int cols, int dims, bool GPU>
float Matrix<rows, cols, dims, GPU>::get(const int _rows, const int _cols) const {
#if SAFE
    if (rows_ >= this->rows || cols_ >= this->cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    if constexpr (GPU) {
        float val;
        checkCUDA(cudaMemcpy(&val, data_d + _rows * this->GetCols() + _cols, sizeof(float), cudaMemcpyDeviceToHost));
        return val;
    }

    return data[_rows * this->GetCols() + _cols];
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::set(const int _rows, const int _cols, const float value) {
#if SAFE
    if (rows_ >= this->rows || cols_ >= this->cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif

    if constexpr (GPU) {
        checkCUDA(cudaMemcpy(data_d + _rows * this->GetCols() + _cols, &value, sizeof(float), cudaMemcpyHostToDevice));
    } else { data[_rows * this->GetCols() + _cols] = value; }
}

template<int rows, int cols, int dims, bool GPU>
template<int other_rows, int other_cols>
void Matrix<rows, cols, dims, GPU>::MatrixMultiplication(const Matrix<other_rows, other_cols> *other,
                                                         Matrix<rows, other_cols> *output) const {
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

    if constexpr (GPU) {
        checkCUBLAS(cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, other->GetCols(), GetRows(), GetCols(),
            &cuda->one, other->data_d, other->GetCols(), data_d, GetCols(), &cuda->zero, output->data_d,
            other->GetCols()));
    } else {
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


        for (int i = 0; i < this->GetRows(); i++) {
            for (int j = 0; j < other->GetCols(); j++) {
                __m128 sum = _mm_setzero_ps();
                int k;
                for (k = 0; k <= this->GetCols() - 4; k += 4) {
                    __m128 a = _mm_loadu_ps(&this->data[i * this->GetCols() + k]);
                    __m128 b = _mm_loadu_ps(&other->data[k * other->GetCols() + j]);
                    sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
                }

                float temp[4];
                _mm_storeu_ps(temp, sum);
                output->data[i * output->GetCols() + j] = temp[0] + temp[1] + temp[2] + temp[3];

                // Handle the remaining elements if cols is not a multiple of 4
                for (; k < this->GetCols(); ++k) {
                    output->data[i * output->GetCols() + j] += this->data[i * this->GetCols() + k] * other->data[k *
                        other->
                        GetCols() + j];
                }
            }
        }
    }
}


template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::CrossProductWithSelfTranspose(const Matrix *other, Matrix *output) const {
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
    for (int i = 0; i < this->GetCols(); i++) {
        for (int j = 0; j < other->GetCols(); j++) {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= this->GetRows() - 4; k += 4) {
                __m128 a = _mm_loadu_ps(&this->data[k * this->GetCols() + i]);
                __m128 b = _mm_loadu_ps(&other->data[k * other->GetCols() + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }

            float temp[4];
            _mm_storeu_ps(temp, sum);
            output->data[i * output->GetCols() + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if rows is not a multiple of 4
            for (; k < this->GetRows(); ++k) {
                output->data[i * output->GetCols() + j] += this->data[k * this->GetCols() + i] * other->data[k * other->
                    GetCols() + j];
            }
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::CrossProductWithTranspose(const Matrix *other, Matrix *output) const {
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

    for (int i = 0; i < this->GetRows(); i++) {
        for (int j = 0; j < other->GetRows(); j++) {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= this->GetCols() - 4; k += 4) {
                __m128 a = _mm_loadu_ps(&this->data[i * this->GetCols() + k]);
                __m128 b = _mm_loadu_ps(&other->data[j * other->GetCols() + k]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }

            float temp[4];
            _mm_storeu_ps(temp, sum);
            output->data[i * output->GetCols() + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < this->GetCols(); ++k) {
                output->data[i * output->GetCols() + j] += this->data[i * this->GetCols() + k] * other->data[j * other->
                    GetCols() + k];
            }
        }
    }
}


template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::OptimizedCrossProduct(const Matrix *a, const Matrix *other, Matrix *output) {
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


    for (int i = 0; i < a->GetRows(); i++) {
        for (int j = 0; j < other->GetCols(); j++) {
            output->data[i * other->GetCols() + j] = 0;
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
            output->data[i * output->cols + j] += temp[0];template <int rows, int cols, int dims, bool GPU>
float& Matrix<rows, cols, dims, GPU>::operator[](int index)
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return data[index];
}

template <int rows, int cols, int dims, bool GPU>
const float& Matrix<rows, cols, dims, GPU>::operator[](int index) const
{
#if SAFE
    if (index >= this->rows * this->cols * this->dim)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
#endif


    return this->data[index];
}
#endif


            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < a->GetCols(); ++k) {
                output->data[i * output->GetCols() + j] += a->data[i * a->GetCols() + k] * other->data[k * other->
                    GetRows() + j];
            }
        }
    }
}


//Read and write matrices.
/*
template<int rows, int cols, int dims, bool GPU>
Matrix* Matrix<rows,cols,dims, GPU>::Read(std::ifstream& reader)
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

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows,cols,dims, GPU>::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(&rows), sizeof(int));
    writer.write(reinterpret_cast<char*>(&cols), sizeof(int));
    writer.write(reinterpret_cast<char*>(&dim), sizeof(int));
    for (int i = 0; i < rows * cols * dim; i++)
    {
        writer.write(reinterpret_cast<char*>(data + i), sizeof(float));
    }
}
*/

template<int rows, int cols, int dims, bool GPU>
float Matrix<rows, cols, dims, GPU>::Distance(Matrix *a, Matrix *b) {
#if SAFE
    if (a->cols != b->cols || a->rows != b->rows)
    {
        throw std::invalid_argument("Matrices need to have same size to calculate distance !");
    }
#endif
    float res = 0;
    for (int i = 0; i < a->GetCols() * a->GetRows(); i++) {
        res += (a[0][i] - b[0][i]) * (a[0][i] - b[0][i]);
    }
    res = std::sqrt(res);
    return res;
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::Copy() {
    if constexpr (GPU) {
        Matrix *res = new Matrix();
        checkCUDA(cudaMemcpy(res->data_d, data_d, GetSize() * sizeof(float), cudaMemcpyDeviceToDevice));
        return res;
    } else {
        auto *resArray = new float[cols * rows * dims];
        for (int i = 0; i < cols * rows * dims; i++) {
            resArray[i] = data[i];
        }
        return new Matrix<rows, cols, dims>(resArray);
    }
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::CopyWithSameData() {
    return new Matrix<rows, cols, dims, GPU>(this->data);
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::Flip180(const Matrix *input, Matrix *output) {
    for (int i = 0; i < input->GetCols() / 2; ++i) {
        for (int j = 0; j < input->GetRows() / 2; ++j) {
            //UGLY
            output->set(i, j, input->get(input->GetRows() - 1 - j, input->GetCols() - 1 - i));
        }
    }
}


template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::GoToNextMatrix() const {
    data += GetMatrixSize();
    offset += GetMatrixSize();
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::ResetOffset() const {
    data -= offset;
    offset = 0;
}

template<int rows, int cols, int dims, bool GPU>
void Matrix<rows, cols, dims, GPU>::SetOffset(const int offset_) const {
    data += offset_;
    this->offset += offset_;
}

template<int rows, int cols, int dims, bool GPU>
int Matrix<rows, cols, dims, GPU>::GetOffset() const {
    return offset;
}


template<int rows, int cols, int dims, bool GPU>
template<int filter_rows, int filter_cols>
void Matrix<rows, cols, dims, GPU>::FullConvolution(const Matrix<rows, cols> *m,
                                                    const Matrix<filter_rows, filter_cols> *filter,
                                                    Matrix<rows + filter_rows - 1, cols + filter_cols - 1, dims> *
                                                    output) {
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
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            float sum = 0;
            for (int k = 0; k < filterRows; k++) {
                for (int l = 0; l < filterCols; l++) {
                    const int inputRow = i + k - r;
                    const int inputCol = j + l - c;
                    if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols) {
                        sum += m->get(inputRow, inputCol) * filter->get(k, l);
                    }
                }
            }
            output->set(i, j, sum);
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


template<int rows, int cols, int dims, bool GPU>
template<int filterSize, int stride>
void Matrix<rows, cols, dims, GPU>::AveragePool(const Matrix<rows, cols, dims, GPU> *a,
                                                Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride +
                                                    1> *output) {
    const int inputCols = a->GetCols();
    const int inputRows = a->GetRows();
    const int outputCols = (inputCols - filterSize) / stride + 1;
    const int outputRows = (inputRows - filterSize) / stride + 1;

    const int fsSquare = filterSize * filterSize;

    for (int d = 0; d < a->GetDims(); d++) {
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                float sum = 0;
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        const int inputRow = i * stride + k;
                        const int inputCol = j * stride + l;
                        if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                            sum += a->get(inputRow, inputCol);
                    }
                }
                output->set(i, j, sum / fsSquare);
            }
        }
        a->GoToNextMatrix();
        output->GoToNextMatrix();
    }

    a->ResetOffset();
    output->ResetOffset();
}

template<int rows, int cols, int dims, bool GPU>
template<int filterSize, int stride>
void Matrix<rows, cols, dims, GPU>::Convolution(
    const Matrix<rows, cols, dims> *input,
    const Matrix<filterSize, filterSize, dims> *filter,
    Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1, dims> *output) {
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

    for (int i = 0; i < output->GetRows(); i++) {
        for (int j = 0; j < output->GetCols(); j++) {
            float sum = 0;
            for (int k = 0; k < filter->GetRows(); k++) {
                for (int l = 0; l < filter->GetRows(); l++) {
                    sum += input->get(i * stride + k, j * stride + l) * filter->get(k, l);
                }
            }
            output->set(i, j, sum);
        }
    }
}

template<int rows, int cols, int dims, bool GPU>
float Matrix<rows, cols, dims, GPU>::Sum() {
    float res = 0;
    for (int i = 0; i < cols * rows; i++) {
        res += data[i];
    }
    return res;
}

/*template <int rows, int cols, int dims, bool GPU>
float& Matrix<rows, cols, dims, GPU>::operator()(int _rows, int _cols)
{
    if (_rows >= rows || _cols >= cols)
    {
        throw std::out_of_range("Matrix : Index out of bounds");
    }
    return data[_rows * this->GetCols() + _cols];
}*/

template<int rows, int cols, int dims, bool GPU>
bool Matrix<rows, cols, dims, GPU>::operator==(const Matrix other) {
    if (other.GetRows() != this->GetRows() || other.GetCols() != this->GetCols() || other.GetDims() != this->
        GetDims()) {
        return false;
    }

    for (int i = 0; i < this->GetSize(); i++) {
        if (abs(other.data[i] - this->data[i]) > 0.0001f) {
            return false;
        }
    }

    return true;
}

/*
template<int rows, int cols, int dims, bool GPU>
void Matrix<rows,cols,dims, GPU>::Flatten() const
{
    rows *= cols * dim;
    cols = 1;
    dim = 1;
}
*/


/*
template<int rows, int cols, int dims, bool GPU>
void Matrix<rows,cols,dims, GPU>::Reshape(const int rows_, const int cols_, const int dimss) const
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
*/

template<int rows, int cols, int dims, bool GPU>
constexpr int Matrix<rows, cols, dims, GPU>::GetSize() const {
    return GetMatrixSize() * dims;
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::Copy(const Matrix *a) {
    auto *res = new Matrix<a->GetRows(), a->GetCols(), a->GetDims()>();
    for (int i = 0; i < a->GetSize(); i++) {
        res[0][i] = a[0][i];
    }
    return res;
}

template<int rows, int cols, int dims, bool GPU>
template<int filterSize, int stride>
void Matrix<rows, cols, dims, GPU>::MaxPool(const Matrix<rows, cols, dims, GPU> *a,
                                            Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1> *
                                            output) {
    const int inputCols = a->GetCols();
    const int inputRows = a->GetRows();
    const int outputCols = (inputCols - filterSize) / stride + 1;
    const int outputRows = (inputRows - filterSize) / stride + 1;

    for (int d = 0; d < a->GetDims(); d++) {
        for (int i = 0; i < outputRows; i++) {
            for (int j = 0; j < outputCols; j++) {
                float max = -DBL_MAX;
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        const int inputRow = i * stride + k;
                        const int inputCol = j * stride + l;
                        if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
                            max = std::max(max, a->get(inputRow, inputCol));
                    }
                }
                output->set(i, j, max);
            }
        }
        a->GoToNextMatrix();
        output->GoToNextMatrix();
    }
    a->ResetOffset();
    output->ResetOffset();
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> Matrix<rows, cols, dims, GPU>::Random() {
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        res[i] = (float) rand() / RAND_MAX * 2 - 1;

    return res;
}

template<int rows, int cols, int dims, bool GPU>
float *Matrix<rows, cols, dims, GPU>::GetData() const {
    if constexpr (GPU)
        return data_d;
    return data;
}

template<int rows, int cols, int dims, bool GPU>
Matrix<rows, cols, dims, GPU> *Matrix<rows, cols, dims, GPU>::operator/=(const float other) {
    for (int i = 0; i < rows * cols * dims; i++) {
        data[i] /= other;
    }
    return this;
}

template<int rows, int cols, int dims, bool GPU>
bool Matrix<rows, cols, dims, GPU>::IsNull(const Matrix *matrix) {
    bool isNull = true;
    for (int i = 0; i < matrix->GetSize(); i++) {
        if ((*matrix)[i] != 0) {
            isNull = false;
            break;
        }
    }
    return isNull;
}

template<int rows, int cols, int dims, bool GPU>
template<int other_rows>
void Matrix<rows, cols, dims, GPU>::MultiplyByTransposeAndAddToRes(const Matrix<other_rows, cols> &other,
                                                                   Matrix<rows, other_rows> &res) requires(GPU) {
    checkCUBLAS(
        cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, other.GetRows(), rows, cols, &cuda->one,
            other.data_d, other.GetCols(), data_d, cols, &cuda->one, res.data_d, other.GetRows()));
}

template<int rows, int cols, int dims, bool GPU>
template<int other_cols>
void Matrix<rows, cols, dims, GPU>::MultiplyTransposeBy(const Matrix<rows, other_cols> &other,
                                                        Matrix<cols, other_cols> &res) requires(GPU) {
    checkCUBLAS(
        cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, other.GetCols(), cols, other.GetRows(), &cuda->one,
            other.data_d, other.GetCols(), data_d, cols, &cuda->zero, res.data_d, other.GetCols()));
}

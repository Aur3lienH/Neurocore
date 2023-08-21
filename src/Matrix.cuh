#pragma once

#include <iostream>
#include <fstream>
#include "Tools/Serializer.cuh"

#define USE_GPU 1

class Matrix
{
public:
    Matrix();

    Matrix(int rows, int cols);

    Matrix(int rows, int cols, int size, bool aligned = false);

    Matrix(int rows, int cols, float value);

    Matrix(int rows, int cols, float* data);

    Matrix(int rows, int cols, int dims, float* data);

    Matrix* Transpose() const;

    virtual ~Matrix();

    static void Flip180(const Matrix* input, Matrix* output);

    static void FullConvolution(const Matrix* m, const Matrix* filter, Matrix* output);

    static void Convolution(const Matrix* a, const Matrix* b, Matrix* output, int stride = 1);

    static void MaxPool(const Matrix* a, Matrix* output, int filterSize = 2, int stride = 2);

    static void AveragePool(const Matrix* a, Matrix* output, int filterSize = 2, int stride = 2);

    static Matrix Random(int rows, int cols);


    //Movement threw the matrix with the offset
    void GoToNextMatrix() const;

    void ResetOffset() const;

    void SetOffset(int offset_) const;

    int GetOffset() const;

    float* GetData() const;


    void Flatten() const;

    void Reshape(int rows_, int cols_, int dims) const;

    void Add(Matrix* other, Matrix* result);

    void AddAllDims(Matrix* other, Matrix* result);

    void Substract(const Matrix* other, Matrix* result) const;

    void SubstractAllDims(const Matrix* other, Matrix* result) const;

    void MultiplyAllDims(const Matrix* other, Matrix* result) const;

    void MultiplyAllDims(float value);

    void DivideAllDims(float value);

    void Zero();

    float Sum();

    int GetRows() const;

    int GetCols() const;

    int GetDims() const;

    int GetSize() const;

    Matrix* operator+=(const Matrix& other);

    Matrix* operator-=(const Matrix& other);

    Matrix* operator+(const Matrix& other) const;

    Matrix* operator*=(const Matrix* other);

    Matrix* operator*=(float other);

    Matrix* operator/=(float other);

    Matrix* operator*(const float& other);

    static Matrix* Read(std::ifstream& reader);

    void Save(std::ofstream& writer);

    float& operator[](int index);

    float& operator()(int rows, int cols);

    const float& operator[](int index) const;

    const float& operator()(int rows, int cols) const;

    const float& operator()(int rows, int cols, int dim) const;

    void CrossProduct(const Matrix* other, Matrix* output) const;


    friend std::ostream& operator<<(std::ostream&, const Matrix&);

    void PrintSize() const;

    static float Distance(Matrix* a, Matrix* b);

    Matrix* Copy();

    Matrix* CopyWithSameData();

    static Matrix* Copy(const Matrix* a);

    //static void FullConvolutionAVX2(const Matrix* m, const Matrix* filter, Matrix* output);

protected:
    mutable float* data;
    mutable int rows;
    mutable int cols;
    mutable int dim;
    mutable int matrixSize;
    mutable int offset = 0;
};


class MatrixCarre : public Matrix
{
public:
    explicit MatrixCarre(int size);

    MatrixCarre(int size, float value);

private:
};


class MatrixDiagonale : public Matrix
{
public:
    MatrixDiagonale(int size, float value);

    ~MatrixDiagonale();
};

/// @brief Same as Matrix but with a different destructor: it doesn't delete the data because it's a clone, the data
/// is deleted by the original matrix
class CloneMatrix : public Matrix
{
public:
    ~CloneMatrix() override = default;

    CloneMatrix() : Matrix()
    {};

    CloneMatrix(int rows, int cols) : Matrix(rows, cols)
    {};

    CloneMatrix(int rows, int cols, int size) : Matrix(rows, cols, size)
    {};

    CloneMatrix(int rows, int cols, float value) : Matrix(rows, cols, value)
    {};

    CloneMatrix(int rows, int cols, float* data) : Matrix(rows, cols, data)
    {};

    CloneMatrix(int rows, int cols, int dims, float* data) : Matrix(rows, cols, dims, data)
    {};
};

#if USE_GPU

#include "CUDA.cuh"

class Matrix_GPU
{
public:
    Matrix_GPU() = default;

    Matrix_GPU(int rows, int cols, int dims = 1);

    explicit Matrix_GPU(const Matrix& mat);

    void Zero();

    void DivideAllDims(float factor);

    virtual ~Matrix_GPU();

    float* GetData_CPU() const;

    // This is a true Matrix multiplication (not Hadamard product)
    static void Multiply(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res);

    static void HadamardProduct(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res);

    void Add(const Matrix_GPU& other, Matrix_GPU& res);

    Matrix_GPU* operator*=(float n);

    void Reshape(int rows_, int cols_, int dims) const;

    void Flatten() const;

    void SetAt(int index, float value);

    float GetAt(int index) const;

    [[nodiscard]] int GetRows() const;

    [[nodiscard]] int GetCols() const;

    [[nodiscard]] int GetDims() const;

    [[nodiscard]] int GetSize() const;

    [[nodiscard]] float* GetData() const;

    Matrix_GPU* Copy() const;

    void Save(std::ofstream& writer) const;

    Matrix_GPU* CopyWithSameData() const;

    [[nodiscard]] cudnnTensorDescriptor_t* GetDescriptor() const;

    [[nodiscard]] cudnnTensorDescriptor_t* GetDescriptor_1D() const;

    static inline CUDA* cuda = new CUDA();

    friend std::ostream& operator<<(std::ostream&, const Matrix_GPU&);

protected:
    float* data_d;
    mutable int rows, cols, dims, size, matrixSize;
    mutable cudnnTensorDescriptor_t desc;
    // Descriptor for the matrix to perform operations on a single dimension
    mutable cudnnTensorDescriptor_t desc_1D;
};

class CloneMatrix_GPU : public Matrix_GPU
{
public:
    ~CloneMatrix_GPU() override
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(desc));
        checkCUDNN(cudnnDestroyTensorDescriptor(desc_1D));
    };

    CloneMatrix_GPU() : Matrix_GPU()
    {};

    CloneMatrix_GPU(int rows, int cols) : Matrix_GPU(rows, cols)
    {};

    CloneMatrix_GPU(int rows, int cols, int size) : Matrix_GPU(rows, cols, size)
    {};
};

#endif


#if USE_GPU
typedef Matrix_GPU MAT;
#else
typedef Matrix MAT;
#endif
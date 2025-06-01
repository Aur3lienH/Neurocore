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

#define AVX2 false
#define SSE2 false

namespace py = pybind11;

template<int rows = 1, int cols = 1, int dims = 1, bool GPU = GPU_DEFAULT>
class Matrix final {
public:
    Matrix(std::initializer_list<float> values);

    Matrix(py::array_t<float> &input);

    static py::object ConvertToArray(py::array_t<float> &input);

    py::array_t<float> ToNumpy();


    explicit Matrix();

    explicit Matrix(float value);

    explicit Matrix(float *newArray, bool owner = false);

    ~Matrix();

    void GPU_init() requires(GPU);

public:
    static void Flip180(const Matrix *input, Matrix *output);

    template<int filter_rows, int filter_cols, int dim1, int dim2, int dim3>
    static void FullConvolution(const Matrix<rows,cols,dim1>* m, const Matrix<filter_rows,filter_cols,dim2>* filter, Matrix<rows+filter_rows-1,cols+filter_cols-1,dim3>* output);

    //static void FullConvolutionAVX2(const Matrix* m, const Matrix* filter, Matrix* output);

    //FullConvolution FS4 = Filter Size 4
    //static void FullConvolutionFS4(const Matrix* m, const Matrix* filter, Matrix* output);

    template<int filterSize, int stride, int dim1, int dim2, int dim3>
    static void Convolution(const Matrix<rows, cols, dim1>* input,const Matrix<filterSize, filterSize, dim2>* filter,Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1, dim3>* output);



    template<int filterSize, int stride>
    static void MaxPool(const Matrix<rows,cols,dims>* a, Matrix<(rows - filterSize) / stride + 1,(cols - filterSize) / stride + 1, dims>* output);

    template<int filterSize, int stride>
    static void AveragePool(const Matrix<rows, cols, dims, GPU> *a,
                            Matrix<(rows - filterSize) / stride + 1, (cols - filterSize) / stride + 1> *output);

    static Matrix* Random();

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
    //void Reshape(int rows_, int cols_, int dims) const;
    template<int dim1, int dim2>
    void Add(Matrix<rows,cols,dim1>* other, Matrix<rows,cols,dim2>* result);

    void AddAllDims(Matrix *other, Matrix *result);

    void Substract(const Matrix *other, Matrix *result) const;

    void SubstractAllDims(const Matrix *other, Matrix *result) const;

    void MultiplyAllDims(const Matrix *other, Matrix *result) const;

    void MultiplyAllDims(float value);

    void DivideAllDims(float value);

    void Zero();

    float Sum();

    static constexpr int GetRows();

    static constexpr int GetCols();

    static constexpr int GetDims();

    static constexpr int GetSize();

    static constexpr int GetMatrixSize();

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

    std::string ToShape() const;

    template<int other_rows, int other_cols>
    void MatrixMultiplication(const Matrix<other_rows, other_cols> *other, Matrix<rows, other_cols> *output) const;

    void CrossProductWithTranspose(const Matrix *other, Matrix *output) const;

    void CrossProductWithSelfTranspose(const Matrix *other, Matrix *output) const;

    static void OptimizedCrossProduct(const Matrix *a, const Matrix *other, Matrix *output);


    void Print() const;

    void PrintSize() const;

    void Print(size_t dimension) const;

    void PrintAllDims() const;

    static float Distance(Matrix* a, Matrix* b);

    Matrix* Copy();

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
    void CheckValidOffset() const;

    mutable float* data = nullptr;

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


#ifndef TEST
#include "Matrix.hxx"
#endif

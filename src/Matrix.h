#pragma once
#include <iostream>
#include <fstream>
#include "Tools/Serializer.h"


class Matrix
{
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, int size);
    Matrix(int rows, int cols, double value);
    Matrix(int rows, int cols, double* data);
    Matrix(int rows, int cols, int dims, double* data);
    ~Matrix();

    static void Flip180(const Matrix* input, Matrix* output);
    static void FullConvolution(const Matrix* m, const Matrix* filter, Matrix* output);
    static void Convolution(const Matrix* a, const Matrix* b, Matrix* output, int stride = 1);
    static void MaxPool(const Matrix* a, Matrix* output, int filterSize = 2, int stride = 2);
    static void AveragePool(const Matrix* a, Matrix* output, int filterSize = 2, int stride = 2);
    static Matrix Random(int rows, int cols);


    //Movement threw the matrix with the offset
    const void GoToNextMatrix() const;
    void ResetOffset() const;
    void SetOffset(int offset) const;
    int GetOffset() const;

    double* GetData();


    void Flatten() const;
    void Reshape(int rows, int cols, int dims) const;
    void Add(Matrix* other, Matrix* result);
    void AddAllDims(Matrix* other, Matrix* result);
    void Substract(const Matrix* other, Matrix* result) const;
    void SubstractAllDims(const Matrix* other, Matrix* result) const;
    void MultiplyAllDims(const Matrix* other, Matrix* result) const;
    void MultiplyAllDims(double value);
    void DivideAllDims(double value);
    void Zero();
    double Sum();
    const int getRows() const;
    const int getCols() const;
    const int getDim() const;
    const int size() const;
    Matrix* operator+=(const Matrix& other);
    Matrix* operator-=(const Matrix& other);
    Matrix* operator+(const Matrix& other);
    Matrix* operator*=(const Matrix* other);
    Matrix* operator*=(const double other);
    Matrix* operator/=(const double other);
    Matrix* operator*(const double& other);
    static Matrix* Read(std::ifstream& reader);
    void Save(std::ofstream& writer);
    double& operator[](int index);
    double& operator()(int rows,int cols);
    const double& operator[](int index) const;
    const double& operator()(int rows,int cols) const;
    const double& operator()(int rows,int cols, int dim) const;
    const void CrossProduct(const Matrix* other, Matrix* output) const;

    
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    void PrintSize() const;
    static float Distance(Matrix* a, Matrix* b);
    Matrix* Copy();
    Matrix* CopyWithSameData();
    static Matrix* Copy(const Matrix* a);

protected:
    mutable double* data;
    mutable int rows;
    mutable int cols;
    mutable int dim;
    mutable int matrixSize;
    mutable int offset = 0;
};



class MatrixCarre : public Matrix
{
public:
    MatrixCarre(int size);
    MatrixCarre(int size, double value);
private:
};


class MatrixDiagonale : public Matrix
{
public:
    MatrixDiagonale(int size, double value);
    ~MatrixDiagonale();
};



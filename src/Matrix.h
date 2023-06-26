#pragma once
#include <iostream>
#include <fstream>
#include "Tools/Serializer.h"


class Matrix
{
public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);
    Matrix(int rows, int cols, double* data);
    ~Matrix();

    static void Flip180(const Matrix* input, Matrix* output);
    static void FullConvolution(const Matrix* m, const Matrix* filter, Matrix* output);
    static void Convolution(const Matrix* a, const Matrix* b, Matrix* output, int stride = 1);
    static void MaxPool(const Matrix* a, Matrix* output, int filter_size, int stride = 1);

    void Add(Matrix* other, Matrix* result);
    void Substract(const Matrix* other, Matrix* result) const ;
    void Zero();
    double Sum();
    const int getRows() const;
    const int getCols() const;
    Matrix* operator+=(const Matrix& other);
    Matrix* operator-=(const Matrix& other);
    Matrix* operator+(const Matrix& other);
    Matrix* operator*=(const Matrix* other);
    Matrix* operator*=(const double other);
    Matrix* operator*(const double& other);
    static Matrix* Read(std::ifstream& reader);
    void Save(std::ofstream& writer);
    double& operator[](int index);
    double& operator()(int rows,int cols);
    const double& operator[](int index) const;
    const double& operator()(int rows,int cols) const;
    const void CrossProduct(const Matrix* other, Matrix* output) const;
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    void PrintSize();
    static float Distance(Matrix* a, Matrix* b);
    Matrix* Copy();
    static Matrix* Copy(const Matrix* a);

protected:
    double* data;
    int rows;
    int cols;
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



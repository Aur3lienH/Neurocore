#pragma once
#include <iostream>
#include <fstream>
#include "Tools/Serializer.h"


class Matrix
{
public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);
    Matrix(int rows, int cols, double* data);
    ~Matrix();
    void Add(Matrix* other, Matrix* result);
    void Subtract(const Matrix* other, Matrix* result) const ;
    void Zero();
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
    const double& operator[](int index) const;
    const double& operator()(int rows,int cols) const;
    const void CrossProduct(const Matrix* other, Matrix* output) const;
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    void PrintSize();
    static float Distance(Matrix* a, Matrix* b);
    Matrix* Copy();
private:
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



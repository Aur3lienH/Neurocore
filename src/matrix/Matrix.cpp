#include <cmath>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <emmintrin.h>
#include <cstdlib>
#include "matrix/Matrix.cuh"




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




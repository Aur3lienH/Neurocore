#include "MatrixTests.h"
#include "../Matrix.h"
#include "../InitFunc.h"
#include <chrono>
#include <iostream>



bool MatrixTests::CrossProductTest1()
{
    const int matrixSize = 1000;
    Matrix* a = new Matrix(matrixSize,matrixSize);
    Matrix* b = new Matrix(matrixSize,matrixSize);
    
    WeightsInit::HeUniform(1,a);
    WeightsInit::HeUniform(1,b);


    Matrix* c = new Matrix(matrixSize,matrixSize);

    Matrix* d = a->Copy();
    Matrix* e = b->Copy();
    Matrix* f = c->Copy();

    auto start = std::chrono::high_resolution_clock::now();
    
    Matrix::CrossProduct(a,b,c);

    auto end = std::chrono::high_resolution_clock::now();

    double duration = (end - start).count();

    std::cout << "Classic cross product took " << duration << " nanoseconds\n";

    start = std::chrono::high_resolution_clock::now();

    Matrix::OptimizedCrossProduct(d,e,f);

    end = std::chrono::high_resolution_clock::now();

    duration = (end - start).count();

    std::cout << "OptimizedCrossProduct took " << duration << " nanoseconds\n";


    return (*c) == (*f);
    
}
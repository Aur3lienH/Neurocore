#include "test/MatrixTests.h"
#include "matrix/Matrix.cuh"
#include "network/InitFunc.cuh"
#include <chrono>
#include <iostream>



bool MatrixTests::SMIDMatrixTest()
{
    const int matrixSize = 1500;


    Matrix* a = new Matrix(matrixSize,matrixSize,true);
    Matrix* b = new Matrix(matrixSize,matrixSize);
    
    WeightsInit::HeUniform(1,a);
    WeightsInit::HeUniform(1,b);


    Matrix* c = new Matrix(matrixSize,matrixSize);

    Matrix* d = a->Copy();
    Matrix* e = b->Copy();
    Matrix* f = c->Copy();

    auto start = std::chrono::high_resolution_clock::now();
    
    a->MatrixMultiplication(b,c);

    auto end = std::chrono::high_resolution_clock::now();

    double duration = (end - start).count();

    std::cout << "Classic cross product took " << duration << " nanoseconds\n";

    start = std::chrono::high_resolution_clock::now();

    Matrix::OptimizedCrossProduct(d,e,f);

    end = std::chrono::high_resolution_clock::now();

    double optiDuration = (end - start).count();

    std::cout << "OptimizedCrossProduct took " << optiDuration << " nanoseconds\n";

    std::cout << "OptimizedCrossProduct is " << duration/optiDuration << " times faster\n";

    

    //c->Print();

    //b->Print();
    //a->Print();
    //c2->Print();
    //std::cout << (*c);
    //std::cout << (*f);


    return (*c) == (*f);
}


bool MatrixTests::BlockMatrixTest()
{
    const int matrixSize = 512;


    Matrix* a = new Matrix(matrixSize,matrixSize,true);
    Matrix* b = new Matrix(matrixSize,matrixSize);
    
    WeightsInit::HeUniform(1,a);
    WeightsInit::HeUniform(1,b);


    Matrix* c = new Matrix(matrixSize,matrixSize);


    auto start = std::chrono::high_resolution_clock::now();
    
    a->MatrixMultiplication(b,c);

    auto end = std::chrono::high_resolution_clock::now();

    double duration = (end - start).count();


    OptimizedMatrix* a2 = OptimizedMatrix::Copy(a);
    OptimizedMatrix* b2 = OptimizedMatrix::Copy(b);
    OptimizedMatrix* c2 = OptimizedMatrix::Copy(c);
    
    start = std::chrono::high_resolution_clock::now();

    a2->MatrixMultiplication(b2,c2);

    end = std::chrono::high_resolution_clock::now();

    double blockDuration = (end - start).count();

    std::cout << "Block Matrixes took " << blockDuration << " nanoseconds\n";
    std::cout << "Block Matrixes is " << duration/blockDuration << " times faster\n";


    return (*c2) == (*c);

}
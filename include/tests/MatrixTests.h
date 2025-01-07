#pragma once
#include "matrix/Matrix.cuh"

class MatrixTests {
public:
    static bool ExecuteTests();
private:
    static bool TestConstructors();
    static bool TestBasicOperations();
    static bool TestMatrixMultiplication();
    static bool TestConvolution();
    static bool TestPooling();
    static bool TestTranspose();
    static bool TestDimensions();
    static bool TestOperators();
    static bool BlockMatrixTest();
};
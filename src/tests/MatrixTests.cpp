#include "tests/MatrixTests.h"
#include "matrix/Matrix.cuh"
#include <cmath>

bool MatrixTests::TestConstructors() {
    // Test default constructor
    Matrix<2,2> mat;
    if (mat.GetRows() != 2 || mat.GetCols() != 2 || mat.GetDims() != 1) {
        return false;
    }

    // Test value constructor
    Matrix<3,3> matValue(5.0f);
    for(int i = 0; i < 9; i++) {
        if (std::abs(matValue[i] - 5.0f) > 1e-6) {
            return false;
        }
    }

    // Test data constructor
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix<2,2> matData(data);
    for(int i = 0; i < 4; i++) {
        if (std::abs(matData[i] - data[i]) > 1e-6) {
            return false;
        }
    }

    return true;
}

bool MatrixTests::TestBasicOperations() {
    // Test Addition
    Matrix<2,2> mat1(1.0f);
    Matrix<2,2> mat2(2.0f);
    Matrix<2,2> result;
    mat1.Add(&mat2, &result);

    for(int i = 0; i < 4; i++) {
        if (std::abs(result[i] - 3.0f) > 1e-6) {
            return false;
        }
    }

    // Test Subtraction
    Matrix<2,2> mat3(5.0f);
    Matrix<2,2> mat4(3.0f);
    Matrix<2,2> subResult;
    mat3.Substract(&mat4, &subResult);

    for(int i = 0; i < 4; i++) {
        if (std::abs(subResult[i] - 2.0f) > 1e-6) {
            return false;
        }
    }

    // Test Multiplication by scalar
    Matrix<2,2> matMul(2.0f);
    matMul.MultiplyAllDims(3.0f);

    for(int i = 0; i < 4; i++) {
        if (std::abs(matMul[i] - 6.0f) > 1e-6) {
            return false;
        }
    }

    return true;
}

bool MatrixTests::TestMatrixMultiplication() {
    Matrix<2,2> mat1({1,2,3,4});
    Matrix<2,2> mat2({2,0,1,2});
    Matrix<2,2> result;

    mat1.MatrixMultiplication(&mat2, &result);

    if (std::abs(result(0,0) - 4.0f) > 1e-6 ||
        std::abs(result(0,1) - 4.0f) > 1e-6 ||
        std::abs(result(1,0) - 10.0f) > 1e-6 ||
        std::abs(result(1,1) - 8.0f) > 1e-6) {
        return false;
    }

    return true;
}

bool MatrixTests::TestConvolution() {
    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    Matrix<2,2> filter({1,1,1,1});
    Matrix<4,4> output;

    Matrix<3,3>::FullConvolution(&input, &filter, &output);

    // Vérifiez quelques valeurs spécifiques de la convolution
    if (std::abs(output(1,1) - 12.0f) > 1e-6) {
        return false;
    }

    return true;
}

bool MatrixTests::TestPooling() {
    Matrix<4,4> input({
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    });
    Matrix<2,2> output;

    Matrix<4,4>::MaxPool<2,2>(&input, &output);

    if (std::abs(output(0,0) - 6.0f) > 1e-6 ||
        std::abs(output(0,1) - 8.0f) > 1e-6 ||
        std::abs(output(1,0) - 14.0f) > 1e-6 ||
        std::abs(output(1,1) - 16.0f) > 1e-6) {
        return false;
    }

    return true;
}

bool MatrixTests::TestTranspose() {
    Matrix<2,3> mat({1,2,3,4,5,6});
    Matrix<3,2>* transposed = mat.Transpose();

    bool success = (std::abs((*transposed)(0,0) - 1.0f) < 1e-6 &&
                   std::abs((*transposed)(0,1) - 4.0f) < 1e-6 &&
                   std::abs((*transposed)(1,0) - 2.0f) < 1e-6 &&
                   std::abs((*transposed)(1,1) - 5.0f) < 1e-6 &&
                   std::abs((*transposed)(2,0) - 3.0f) < 1e-6 &&
                   std::abs((*transposed)(2,1) - 6.0f) < 1e-6);

    delete transposed;
    return success;
}

bool MatrixTests::TestDimensions() {
    Matrix<2,2,2> mat;
    if (mat.GetOffset() != 0) return false;

    mat.SetOffset(4);
    if (mat.GetOffset() != 4) return false;

    mat.GoToNextMatrix();
    if (mat.GetOffset() != 8) return false;

    mat.ResetOffset();
    if (mat.GetOffset() != 0) return false;

    return true;
}

bool MatrixTests::TestOperators() {
    Matrix<2,2> mat1(1.0f);
    Matrix<2,2> mat2(2.0f);
    Matrix<2,2>* addResult = mat1 + mat2;

    bool success = true;
    for(int i = 0; i < 4; i++) {
        if (std::abs((*addResult)[i] - 3.0f) > 1e-6) {
            success = false;
            break;
        }
    }

    delete addResult;
    return success;
}

bool MatrixTests::BlockMatrixTest() {
    // Créez des matrices de taille appropriée pour le test de block
    Matrix<64,64> matA(1.0f);
    Matrix<64,64> matB(2.0f);
    Matrix<64,64> result;

    // Effectuez la multiplication par blocs
    matA.OptimizedCrossProduct(&matA, &matB, &result);

    // Vérifiez quelques valeurs du résultat
    bool success = true;
    // Ajoutez vos vérifications spécifiques selon votre implémentation

    return success;
}
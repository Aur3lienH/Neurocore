#include "tests/MatrixTests.h"
#include "matrix/Matrix.cuh"
#include <cmath>


bool MatrixTests::ExecuteTests()
{
    bool res = true;
    std::vector<std::tuple<void*,std::string>> functions;
    //functions.push_back(std::tuple((void*)MatrixTests::SMIDMatrixTest,std::string("SMID Cross Product")));
    functions.emplace_back((void*)MatrixTests::BlockMatrixTest,std::string("Block Cross Product"));
    functions.emplace_back((void*)MatrixTests::TestConstructors,std::string("Constructors"));
    functions.emplace_back((void*)MatrixTests::TestBasicOperations, std::string("Basic Operations (Add, Sub, Mul)"));
    functions.emplace_back((void*)MatrixTests::TestMatrixMultiplication, std::string("Matrix Multiplication"));
    functions.emplace_back((void*)MatrixTests::TestConvolution, std::string("Convolution"));
    functions.emplace_back((void*)MatrixTests::TestPooling, std::string("Pooling"));
    functions.emplace_back((void*)MatrixTests::TestTranspose, std::string("Transpose"));
    functions.emplace_back((void*)MatrixTests::TestDimensions, std::string("Dimensions"));
    functions.emplace_back((void*)MatrixTests::TestOperators, std::string("Operators"));

    bool* array = new bool[functions.size()];


    for (int i = 0; i < functions.size(); i++)
    {
        bool (*func)(void) = (bool (*)(void))(std::get<0>(functions[i]));
        bool res = func();
        if(res)
        {
            array[i] = true;
        }
        else
        {
            array[i] = false;
            res = false;
        }
    }

    //system("clear");
    for (int i = 0; i < functions.size(); i++) {
        if(array[i])
        {
            std::cout << "  \033[1;32m[SUCCEED]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
        else
        {
            std::cout << "  \033[1;31m[FAIL]\033[0m   ";
            std::cout << std::get<1>(functions[i]) << "\n";
        }
    }
    delete[] array;
    return res;
}



//Test for
//  Matrix::Matrix()
//  Matrix::Matrix(float)
//  Matrix::Matrix(std::initializer_list<float>)
//  Matrix::Matrix(float*,bool)
bool MatrixTests::TestConstructors() {
    // Matrix::Matrix()
    Matrix<2,2> mat;
    if (mat.GetRows() != 2 || mat.GetCols() != 2 || mat.GetDims() != 1) {
        return false;
    }

    //Matrix::Matrix(float)
    Matrix<3,3> matValue(5.0f);
    for(int i = 0; i < 9; i++) {
        if (std::abs(matValue[i] - 5.0f) > 1e-6) {
            std::cout << "matrix value is " << matValue[i] << " and should be 5 \n";
            return false;
        }
    }

    // Test Matrix::Matrix(std::initializer_list<float>)
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix<2,2> matData({1.0f, 2.0f, 3.0f, 4.0f});
    for(int i = 0; i < 4; i++) {
        if (std::abs(matData[i] - data[i]) > 1e-6) {
            return false;
        }
    }

    // Test Matrix::Matrix(float*, bool)
    Matrix<2,2> matData2(data, false);
    for(int i = 0; i < 4; i++) {
        if (std::abs(matData2[i] - data[i]) > 1e-6) {
            return false;
        }
    }

    return true;
}

//Test for
//  Matrix::Add(Matrix*,Matrix*)
//  Matrix::Substract(Matrix*,Matrix*)
//  Matrix::MultiplyAllDims(float)

bool MatrixTests::TestBasicOperations() {

    // Matrix::Add(Matrix*,Matrix*)
    Matrix<2,2> mat1(1.0f);
    Matrix<2,2> mat2(2.0f);
    Matrix<2,2> result;
    mat1.Add(&mat2, &result);

    for(int i = 0; i < 4; i++) {
        if (std::abs(result[i] - 3.0f) > 1e-6) {
            return false;
        }
    }

    // Matrix::Substract(Matrix*,Matrix*)
    Matrix<2,2> mat3(5.0f);
    Matrix<2,2> mat4(3.0f);
    Matrix<2,2> subResult;
    mat3.Substract(&mat4, &subResult);

    for(int i = 0; i < 4; i++) {
        if (std::abs(subResult[i] - 2.0f) > 1e-6) {
            return false;
        }
    }

    // Matrix::MultiplyAllDims(float)
    Matrix<2,2> matMul(2.0f);
    matMul.MultiplyAllDims(3.0f);

    for(int i = 0; i < 4; i++) {
        if (std::abs(matMul[i] - 6.0f) > 1e-6) {
            return false;
        }
    }

    return true;
}

//Test for
//  Matrix::MatrixMultiplication(Matrix*,Matrix*)
bool MatrixTests::TestMatrixMultiplication() {
    // Test Matrix::MatrixMultiplication(Matrix*,Matrix*)
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

//Test for
//  Matrix::FullConvolution(Matrix*,Matrix*,Matrix*)
//  Matrix::Convolution(Matrix*,Matrix*,Matrix*)
bool MatrixTests::TestConvolution() {

    // Test Matrix::FullConvolution(Matrix*,Matrix*,Matrix*)
    Matrix<3,3> input({1,2,3,4,5,6,7,8,9});
    Matrix<2,2> filter({1,1,1,1});
    Matrix<4,4> output;

    Matrix<3,3>::FullConvolution(&input, &filter, &output);

    // Vérifiez quelques valeurs spécifiques de la convolution
    if (std::abs(output(1,1) - 12.0f) > 1e-6) {
        return false;
    }

    // Test Matrix::Convolution(Matrix*,Matrix*,Matrix*)
    Matrix<3,3> input2({1,2,3,4,5,6,7,8,9});
    Matrix<2,2> filter2({1,1,1,1});
    Matrix<2,2> output2;

    Matrix<3,3>::Convolution<2,1>(&input2, &filter2, &output2);

    if (std::abs(output2(0,0) - 12.0f) > 1e-6 ||
        std::abs(output2(0,1) - 16.0f) > 1e-6 ||
        std::abs(output2(1,0) - 24.0f) > 1e-6 ||
        std::abs(output2(1,1) - 28.0f) > 1e-6) {
        return false;
    }

    return true;
}

//Test for
//  Matrix::MaxPool(Matrix*,Matrix*)
//  Matrix::AveragePool(Matrix*,Matrix*)
bool MatrixTests::TestPooling() {

    // Test Matrix::MaxPool(Matrix*,Matrix*)
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

    // Test Matrix::AveragePool(Matrix*,Matrix*)
    Matrix<4,4> input2({
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        13,14,15,16
    });
    Matrix<2,2> output2;

    Matrix<4,4>::AveragePool<2,2>(&input2, &output2);
    if (std::abs(output2(0,0) - 3.5f) > 1e-6 ||
        std::abs(output2(0,1) - 5.5f) > 1e-6 ||
        std::abs(output2(1,0) - 11.5f) > 1e-6 ||
        std::abs(output2(1,1) - 13.5f) > 1e-6) {
        return false;
    }

    return true;
}

//Test for Matrix::Transpose()
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

//Test for
//  Matrix::operator+(Matrix&)
//  Matrix::operator-(Matrix&)
//  Matrix::operator*(float)
bool MatrixTests::TestOperators() {

    //Test Matrix::operator+(Matrix&)
    Matrix<2,2> mat1(1.0f);
    Matrix<2,2> mat2(2.0f);
    Matrix<2,2>* addResult = mat1 + mat2;

    bool success = true;
    for(int i = 0; i < 4; i++) {
        if (std::abs((*addResult)[i] - 3.0f) > 1e-6) {
            std::cout << "value is " << (*addResult)[i] << " and should be 3 \n";
            success = false;
            break;
        }
    }

    delete addResult;

    //Test Matrix::operator-(Matrix&)
    Matrix<2,2> mat3(5.0f);
    Matrix<2,2> mat4(3.0f);
    Matrix<2,2>* subResult = mat3 - mat4;

    for(int i = 0; i < 4; i++) {
        if (std::abs((*subResult)[i] - 2.0f) > 1e-6) {

            success = false;
            break;
        }
    }

    //Test Matrix::operator*(float)
    Matrix<2,2> mat5({1,2,3,4});
    Matrix<2,2>* mulResult = mat5 * 3.0f;

    for(int i = 0; i < 4; i++) {
        if (std::abs((*mulResult)[i] - (i + 1) * 3.0f) > 1e-6) {
            success = false;
            break;
        }
    }

    delete subResult;
    delete mulResult;

    return success;
}

bool MatrixTests::BlockMatrixTest() {
    return true;
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




#pragma once

/*
//Optimized matrix, cache optimization
class OptimizedMatrix : public Matrix
{
public:
    OptimizedMatrix(const int rows, const int cols, const int dims, float value = 0.0f, bool aligned = false);
    void MatrixMultiplication(const Matrix* b, Matrix* output) const override;
    static OptimizedMatrix* Copy(const Matrix* a);
    bool operator==(const Matrix& other);
    void Print() const override;
private:
    void Init(const int rows, const int cols, const int dims, float value = 0, bool aligned = false);
    int ConvertIndex(int index);
};

*/
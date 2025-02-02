#pragma once
/*

class HorizontalMatrix : public Matrix
{
public:
    HorizontalMatrix(const int rows, const int cols, const int dims, float value = 0.0f, bool aligned = false);

private:
    void Init(const int rows, const int cols, const int dims, float value = 0, bool aligned = false);
};


class MatrixCarre : public Matrix
{
public:
    explicit MatrixCarre(int size);

    MatrixCarre(int size, float value);

private:
};


class MatrixDiagonale : public Matrix
{
public:
    MatrixDiagonale(int size, float value);

    ~MatrixDiagonale();
};

/// @brief Same as Matrix but with a different destructor: it doesn't delete the data because it's a clone, the data
/// is deleted by the original matrix
class CloneMatrix : public Matrix
{
public:
    ~CloneMatrix() override
    {};

    CloneMatrix() : Matrix()
    {};

    CloneMatrix(int rows, int cols) : Matrix(rows, cols)
    {};

    CloneMatrix(int rows, int cols, int size) : Matrix(rows, cols, size)
    {};

    CloneMatrix(int rows, int cols, float value) : Matrix(rows, cols, value)
    {};

    CloneMatrix(int rows, int cols, float* data) : Matrix(rows, cols, data)
    {};

    CloneMatrix(int rows, int cols, int dims, float* data) : Matrix(rows, cols, dims, data)
    {};
};
*/
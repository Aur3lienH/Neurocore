#pragma once



#if USE_GPU

#include "CUDA.cuh"

template<
class Matrix_GPU
{
public:
    Matrix_GPU() = default;

    Matrix_GPU(int rows, int cols, int dims = 1);

    explicit Matrix_GPU(const Matrix& mat);

    void Zero();

    void DivideAllDims(float factor);

    virtual ~Matrix_GPU();

    float* GetData_CPU() const;

    float* GetData_CPU_1D() const;

    // This is a true Matrix multiplication (not Hadamard product)
    static void Multiply(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res);

    void MultiplyByTransposeAndAddToRes(const Matrix_GPU& other, Matrix_GPU& res);

    void MultiplyTransposeBy(const Matrix_GPU& other, Matrix_GPU& res);

    static void HadamardProduct(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res);

    void Add(const Matrix_GPU& other, Matrix_GPU& res);

    Matrix_GPU* operator*=(float n);

    void Reshape(int rows_, int cols_, int dims) const;

    void Flatten() const;

    void SetAt(int index, float value);

    float GetAt(int index) const;

    [[nodiscard]] int GetRows() const;

    [[nodiscard]] int GetCols() const;

    [[nodiscard]] int GetDims() const;

    [[nodiscard]] int GetSize() const;

    [[nodiscard]] float* GetData() const;

    [[nodiscard]] int GetMatrixSize() const;

    Matrix_GPU* Copy() const;

    void Save(std::ofstream& writer) const;

    Matrix_GPU* CopyWithSameData() const;

    static inline CUDA* cuda = new CUDA();

    friend std::ostream& operator<<(std::ostream&, const Matrix_GPU&);

    static void DisplayTensorInfo(const cudnnTensorDescriptor_t& desc);

protected:
    float* data_d;
    mutable int rows, cols, dims, size, matrixSize, offset;
    mutable cudnnTensorDescriptor_t desc;
    // Descriptor for the matrix to perform operations on a single dimension
    mutable cudnnTensorDescriptor_t desc_1D;
};

class CloneMatrix_GPU : public Matrix_GPU
{
public:
    ~CloneMatrix_GPU() override
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(desc));
        checkCUDNN(cudnnDestroyTensorDescriptor(desc_1D));
    };

    CloneMatrix_GPU() : Matrix_GPU()
    {};

    CloneMatrix_GPU(int rows, int cols) : Matrix_GPU(rows, cols)
    {};

    CloneMatrix_GPU(int rows, int cols, int size) : Matrix_GPU(rows, cols, size)
    {};
};

#endif
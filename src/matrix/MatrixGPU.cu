#if USE_GPU

#include "matrix/Matrix.cuh"
#include "CUDA.h"


Matrix_GPU::Matrix_GPU(const int rows, const int cols, const int dims) : rows(rows), cols(cols), dims(dims),
                                                                         matrixSize(rows * cols),
                                                                         size(rows * cols * dims), offset(0)
{
    checkCUDA(cudaMalloc(&data_d, rows * cols * dims * sizeof(float)));
    checkCUDA(cudaMemset(data_d, 0, rows * cols * dims * sizeof(float)));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_1D));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
}

Matrix_GPU::Matrix_GPU(const Matrix& mat) : Matrix_GPU(mat.GetRows(), mat.GetCols(), mat.GetDims())
{
#if SAFE
    if (size != mat.GetSize())
        throw std::runtime_error("Matrices dimensions do not match !");
#endif
    checkCUDA(cudaMemcpy(data_d, mat.GetData(), mat.GetSize() * sizeof(float), cudaMemcpyHostToDevice));
}

Matrix_GPU::~Matrix_GPU()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(desc_1D));
    cudaFree(data_d);
}

void Matrix_GPU::Zero()
{
    checkCUDA(cudaMemset(data_d, 0, size * sizeof(float)));
}

float* Matrix_GPU::GetData_CPU() const
{
    float* data_h = new float[size];
    checkCUDA(cudaMemcpy(data_h, data_d - offset, size * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

float* Matrix_GPU::GetData_CPU_1D() const
{
    float* data_h = new float[matrixSize];
    checkCUDA(cudaMemcpy(data_h, data_d, matrixSize * sizeof(float), cudaMemcpyDeviceToHost));

    return data_h;
}

void Matrix_GPU::DivideAllDims(const float factor)
{
    const float multFactor = 1.f / factor;
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc, data_d, &multFactor));
}

void Matrix_GPU::Multiply(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res)
{
    // Help to deal with CUBLAS fucking column-major order
    //https://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
    //https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp#L293
    checkCUBLAS(cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, b.GetCols(), a.GetRows(), a.GetCols(),
                               &cuda->one, b.data_d, b.GetCols(), a.data_d, a.GetCols(), &cuda->zero, res.data_d,
                               b.GetCols()));
}

void Matrix_GPU::MultiplyByTransposeAndAddToRes(const Matrix_GPU& other, Matrix_GPU& res)
{
    checkCUBLAS(
            cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, other.rows, rows, cols, &cuda->one,
                           other.data_d, other.cols, data_d, cols, &cuda->one, res.data_d, other.rows));
}

void Matrix_GPU::MultiplyTransposeBy(const Matrix_GPU& other, Matrix_GPU& res)
{
    /*int values[] = {rows, cols, other.rows, other.cols};
    for (const int m : values)
    {
        for (const int n : values)
        {
            for (const int k : values)
            {
                for (const int lda : values)
                {
                    for (const int ldb : values)
                    {
                        for (const int ldc : values)
                        {
                            std::cout << "M: " << m << ", N: " << n << ", K: " << k << ", ldA: " << lda << ", ldB: "
                                      << ldb << ", ldC: " << ldc << std::endl;
                            try
                            {
                                checkCUBLAS(
                                        cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                                       &cuda->one,
                                                       other.data_d, lda, data_d, ldb, &cuda->zero, res.data_d,
                                                       ldc));
                                std::cout << "Success !\n";
                                if (std::abs(res.GetAt(0) - 0.0109781f) < .01)
                                    std::cout << res << std::endl << std::flush;
                            }
                            catch (std::exception& e)
                            {
                                std::cout << "Fail" << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }*/
    checkCUBLAS(
            cublasSgemm_v2(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, other.cols, cols, other.rows, &cuda->one,
                           other.data_d, other.cols, data_d, cols, &cuda->zero, res.data_d, other.cols));
}

void Matrix_GPU::Add(const Matrix_GPU& other, Matrix_GPU& res)
{
#if SAFE
    if (matrixSize != other.matrixSize)
        throw std::runtime_error("Matrices dimensions mismatch");
#endif
    checkCUBLAS(
            cublasSgeam(cuda->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols,
                        &cuda->one, data_d, rows, &cuda->one, other.data_d, rows, res.data_d, rows));
}

Matrix_GPU* Matrix_GPU::operator*=(const float n)
{
    //checkCUBLAS(cublasSscal_v2(cuda->cublasHandle, matrixSize, &n, data_d, 1));
    checkCUDNN(cudnnScaleTensor(cuda->cudnnHandle, desc_1D, data_d, &n));

    return this;
}

#include "tools/Serializer.h"
void Matrix_GPU::Reshape(int rows_, int cols_, int dims_) const
{
    rows = rows_;
    cols = cols_;
    dims = dims_;
    checkCUDNN(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, dims, rows, cols));
    checkCUDNN(cudnnSetTensor4dDescriptor(desc_1D, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, rows, cols));
}

void Matrix_GPU::Flatten() const
{
    Reshape(size, 1, 1);
}

float Matrix_GPU::GetAt(const int index) const
{
    float res;
    checkCUDA(cudaMemcpy(&res, data_d + index, sizeof(float), cudaMemcpyDeviceToHost));

    return res;
}

void Matrix_GPU::SetAt(const int index, const float value)
{
    checkCUDA(cudaMemcpy(data_d + index, &value, sizeof(float), cudaMemcpyHostToDevice));
}

int Matrix_GPU::GetRows() const
{
    return rows;
}

int Matrix_GPU::GetCols() const
{
    return cols;
}

int Matrix_GPU::GetDims() const
{
    return dims;
}

int Matrix_GPU::GetSize() const
{
    return size;
}

float* Matrix_GPU::GetData() const
{
    return data_d;
}

int Matrix_GPU::GetMatrixSize() const
{
    return matrixSize;
}

Matrix_GPU* Matrix_GPU::Copy() const
{
    auto* res = new Matrix_GPU(rows, cols, dims);
    checkCUDA(cudaMemcpy(res->data_d, data_d, size * sizeof(float), cudaMemcpyDeviceToDevice));
    return res;
}

void Matrix_GPU::Save(std::ofstream& writer) const
{
    Matrix cpy(rows, cols, GetData_CPU());
    cpy.Save(writer);
}

Matrix_GPU* Matrix_GPU::CopyWithSameData() const
{
    CloneMatrix_GPU* res = new CloneMatrix_GPU(rows, cols, dims);
    res->data_d = data_d;

    return res;
}

__global__
void CoeffWiseMultKernel(const float* a, const float* b, float* res, const int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        res[index] = a[index] * b[index];

}

void Matrix_GPU::HadamardProduct(const Matrix_GPU& a, const Matrix_GPU& b, Matrix_GPU& res)
{
    std::cout << "This is done on all Dims\n";
#if SAFE
    if (a.GetSize() != b.GetSize())
        throw std::runtime_error("Matrices dimensions mismatch !");
#endif
    const int size = a.GetSize();
    const int blocksPerGrid = (size + cuda->threadsPerBlock - 1) / cuda->threadsPerBlock;
    checkKernel((CoeffWiseMultKernel<<<blocksPerGrid, cuda->threadsPerBlock>>>(a.data_d, b.data_d, res.data_d, size)));
    checkCUDA(cudaDeviceSynchronize());
}

std::ostream& operator<<(std::ostream& os, const Matrix_GPU& matrix)
{
    std::cout << "Matrix: " << matrix.rows << "x" << matrix.cols << std::endl;
    float* data = matrix.GetData_CPU() + matrix.offset;
    for (int i = 0; i < matrix.rows; i++)
    {
        os << "[";
        for (int j = 0; j < matrix.cols; j++)
        {
            os << data[i * matrix.cols + j];
            if (j != matrix.cols - 1)
            {
                os << " ";
            }
        }
        os << "]\n";
    }

    data -= matrix.offset;
    delete data;

    return os;
}

void Matrix_GPU::DisplayTensorInfo(cudnnTensorDescriptor_t const& desc)
{
    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;
    checkCUDNN(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    std::cout << "DataType: " << dataType << ", n: " << n << ", c: " << c << ", h: " << h << ", w: " << w
              << ", nStride: "
              << nStride << ", cStride: " << cStride << ", hStride: " << hStride << ", wStride: " << wStride
              << std::endl;
}


#endif
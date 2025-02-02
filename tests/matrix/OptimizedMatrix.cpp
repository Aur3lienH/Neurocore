/*
#include "matrix/Matrix.cuh"
#include <emmintrin.h>
#include <immintrin.h>
#include "network/Allocator.h"

//The size of the blocks which constitue the matrix
const int BLOCK_SIZE = 4;

#define AVX2 false
#define SSE2 true


void OptimizedMatrix::Init(int rows, int cols, int dims, float value, bool aligned)
{
    this->rows = rows;
    this->cols = cols;
    this->dim = dims;
    this->matrixSize = rows * cols;
    this->data = new float[matrixSize * dims];
    if (aligned)
    {
        if (posix_memalign((void**)&this->data, 16, sizeof(float) * matrixSize * dims))
        {
            throw std::invalid_argument("Cannot create an aligned array ! ");
        }
    }
    for (int i = 0; i < matrixSize * dims; i++)
    {
        this->data[i] = value;
    }
}



OptimizedMatrix::OptimizedMatrix(const int rows, const int cols, const int dims, float value, bool aligned)
{
    Init(rows, cols, dims, value, aligned);
}


//Divide matrices into blocks
void OptimizedMatrix::MatrixMultiplication(const Matrix* b, Matrix* output) const
{
    float* left = this->GetData();
    float* right = b->GetData();
    float* out = output->GetData();
    

    int blockSizeSquared = BLOCK_SIZE * BLOCK_SIZE;
    int rowsRightBlockCount = b->GetRows() / BLOCK_SIZE;
    int colsRightBlockCount = b->GetCols() / BLOCK_SIZE;
    output->Zero();
    //The two loops are used to go through the blocks of the matrices
    int counter = 0;
    for (int f = 0; f < colsRightBlockCount; f++)
    {
        right = b->GetData();
        for(int i = 0; i < rowsRightBlockCount; i++)
        {
            for (int j = 0; j < colsRightBlockCount; j++)
            {
            */
                /*
                std::cout << "left : " << left - this->GetData() << "\n";
                std::cout << "rigth : " << right - b->GetData() << "\n";
                std::cout << "out : " << out - output->GetData() << "\n";
                std::cout << "\n";
                */
/*
                for (int ii = 0; ii < BLOCK_SIZE; ii++)
                {
                    //Go to the next row in this matrix and go to the next column in the other matrix
                    for (int jj = 0; jj < BLOCK_SIZE; jj++)
                    {
                        float* res = (out + jj * BLOCK_SIZE + ii);
                        //Go to the next column in this matrix and go to the next rows in the other matrix
                        int kk = 0;
                        */
                        /*
                        __m128 sum = _mm_setzero_ps();
                        for(;kk + 4 <= BLOCK_SIZE; kk+=4)
                        {
                            __m128 m_a = _mm_loadu_ps(left + kk + jj * BLOCK_SIZE);

                            __m128 m_b = _mm_set_ps(right[(kk) * BLOCK_SIZE + ii], right[(kk + 1) * BLOCK_SIZE + ii],
                                        right[(kk + 2) * BLOCK_SIZE + ii], right[(kk + 3) * BLOCK_SIZE + ii]);

                            sum = _mm_add_ps(sum, _mm_mul_ps(m_a, m_b));
                            
                            _mm_storeu_ps(Allocator::temp, sum);
                            *res += Allocator::temp[0] + Allocator::temp[1] + Allocator::temp[2] + Allocator::temp[3];

                        }
                        */
/*
                        

                        for (; kk < BLOCK_SIZE; kk++)
                        {
                            *res += left[kk + jj * BLOCK_SIZE] * right[kk * BLOCK_SIZE + ii];
                        }
                    }
                }

                right += blockSizeSquared;
                out += blockSizeSquared;
            }
            out -= blockSizeSquared * colsRightBlockCount; 
            left += blockSizeSquared;
        }   
        out += blockSizeSquared * rowsRightBlockCount;

    }
    

    
}


OptimizedMatrix* OptimizedMatrix::Copy(const Matrix* a)
{
    OptimizedMatrix* res = new OptimizedMatrix(a->GetRows(), a->GetCols(), a->GetDims());
    
    if(a->GetCols() % BLOCK_SIZE != 0 || a->GetRows() % BLOCK_SIZE != 0)
    {
        throw new std::invalid_argument("The number of cols and of rows must be a multiple of BLOCKSIZE");
    }

    int blockSquared = BLOCK_SIZE * BLOCK_SIZE;

    for (int i = 0; i < a->GetRows(); i+=BLOCK_SIZE)
    {
        for (int j = 0; j < a->GetCols(); j+=BLOCK_SIZE)
        {
            for (int ii = 0; ii < BLOCK_SIZE; ii++)
            {
                for (int jj = 0; jj < BLOCK_SIZE; jj++)
                {
                    //std::cout << i * a->GetCols() + j * BLOCK_SIZE + jj + ii * BLOCK_SIZE << " = " << (ii * a->GetCols() + jj) + j + i * a->GetCols() << "\n";
                    (*res)[i * a->GetCols() + j * BLOCK_SIZE + jj + ii * BLOCK_SIZE] = (*a)[(ii * a->GetCols() + jj) + j + i * a->GetCols()];
                }
            }
        }
        
    }


    return res;
    
    
}


void OptimizedMatrix::Print() const 
{
    Matrix matrix = *this;
    std::cout << "Matrix: " << matrix.GetRows() << "x" << matrix.GetCols() << std::endl;

    int squareBlockSize = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = 0; i < matrix.GetRows(); i++)
    {
        std::cout << "[";
        for (int j = 0; j < matrix.GetCols(); j++)
        {
            std::cout << matrix[(j / BLOCK_SIZE) * squareBlockSize + j % BLOCK_SIZE + (i%BLOCK_SIZE) * BLOCK_SIZE + (i/BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE * (matrix.GetCols() / BLOCK_SIZE)];
            //std::cout << "print : " << (j / BLOCK_SIZE) * squareBlockSize + j % BLOCK_SIZE + (i%BLOCK_SIZE) * BLOCK_SIZE + (i/BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE * (matrix.GetCols() / BLOCK_SIZE) << "\n";
            if (j != matrix.GetCols() - 1)
            {
                std::cout << " ";
            }
        }
        std::cout << "]\n";
        
    }
    
}


bool OptimizedMatrix::operator==(const Matrix& other)
{
    int squareBlockSize = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = 0; i < this->GetRows(); i++)
    {
        for (int j = 0; j < this->GetCols(); j++)
        {
            float value1 = (*this)[(j / BLOCK_SIZE) * squareBlockSize + j % BLOCK_SIZE + (i%BLOCK_SIZE) * BLOCK_SIZE + (i/BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE * (this->GetCols() / BLOCK_SIZE)];
            float value2 = other[j + i * this->GetCols()];
            if(abs(value1 - value2) > 0.01)
            {
                return false;
            }
        }
        
    }
    return true;
}
*/
/**
 * Standalone C++ test for optimized matrix multiplication
 * Does NOT use pybind11 - only tests the core matrix operations
 */

#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <emmintrin.h>
#include <pmmintrin.h>

using namespace std;
using namespace std::chrono;

// Simplified Matrix class template for testing (without pybind11)
template<int rows, int cols, int dims = 1>
class SimpleMatrix {
public:
    float* data;
    bool owner;
    
    SimpleMatrix() : owner(true) {
        data = new float[rows * cols * dims]();
    }
    
    ~SimpleMatrix() {
        if (owner) {
            delete[] data;
        }
    }
    
    constexpr int GetRows() const { return rows; }
    constexpr int GetCols() const { return cols; }
    constexpr int GetDims() const { return dims; }
    
    float& operator()(int r, int c) {
        return data[r * cols + c];
    }
    
    const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }
    
    float& operator[](int index) {
        return data[index];
    }
    
    const float& operator[](int index) const {
        return data[index];
    }
    
    // Optimized matrix multiplication with cache blocking
    template<int other_rows, int other_cols>
    void MatrixMultiplication(const SimpleMatrix<other_rows, other_cols>* other, SimpleMatrix<rows, other_cols>* output) const {
        constexpr int BLOCK_SIZE = 64;
        constexpr int BLOCK_SIZE_K = 256;
        
        const int M = rows;
        const int N = other_cols;
        const int K = cols;
        
        // Zero output
        for (int i = 0; i < M * N; i++) {
            output->data[i] = 0.0f;
        }
        
        // Cache-friendly blocked multiplication
        for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
            int iend = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
            for (int kk = 0; kk < K; kk += BLOCK_SIZE_K) {
                int kend = (kk + BLOCK_SIZE_K < K) ? kk + BLOCK_SIZE_K : K;
                for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                    int jend = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
                    
                    for (int i = ii; i < iend; i++) {
                        const float* row_a = &data[i * K];
                        float* row_c = &output->data[i * N];
                        
                        for (int j = jj; j < jend; j++) {
                            __m128 sum = _mm_setzero_ps();
                            int k = kk;
                            
                            for (; k <= kend - 4; k += 4) {
                                __m128 a = _mm_loadu_ps(&row_a[k]);
                                __m128 b = _mm_set_ps(
                                    other->data[(k+3) * N + j],
                                    other->data[(k+2) * N + j],
                                    other->data[(k+1) * N + j],
                                    other->data[k * N + j]
                                );
                                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
                            }
                            
                            sum = _mm_hadd_ps(sum, sum);
                            sum = _mm_hadd_ps(sum, sum);
                            float vec_sum;
                            _mm_store_ss(&vec_sum, sum);
                            row_c[j] += vec_sum;
                            
                            for (; k < kend; k++) {
                                row_c[j] += row_a[k] * other->data[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }
};

// Helper to generate random matrix
template<int rows, int cols>
void FillRandom(SimpleMatrix<rows, cols>* mat) {
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        (*mat)[i] = dis(gen);
    }
}

// Naive matrix multiplication for verification
template<int M, int K, int N>
void NaiveMatMul(const SimpleMatrix<M, K>* A, const SimpleMatrix<K, N>* B, SimpleMatrix<M, N>* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (*A)(i, k) * (*B)(k, j);
            }
            (*C)(i, j) = sum;
        }
    }
}

template<int M, int K, int N>
bool TestMatrixMultiplication() {
    cout << "Testing " << M << "x" << K << " @ " << K << "x" << N << "... " << flush;
    
    auto* A = new SimpleMatrix<M, K>();
    auto* B = new SimpleMatrix<K, N>();
    auto* C_optimized = new SimpleMatrix<M, N>();
    auto* C_naive = new SimpleMatrix<M, N>();
    
    FillRandom(A);
    FillRandom(B);
    
    A->MatrixMultiplication(B, C_optimized);
    NaiveMatMul(A, B, C_naive);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = abs((*C_optimized)[i] - (*C_naive)[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    bool passed = (max_diff < 1e-3f);
    
    if (passed) {
        cout << "✓ (max_diff: " << scientific << setprecision(2) << max_diff << ")" << endl;
    } else {
        cout << "✗ (max_diff: " << scientific << setprecision(2) << max_diff << ")" << endl;
    }
    
    delete A;
    delete B;
    delete C_optimized;
    delete C_naive;
    
    return passed;
}

template<int M, int K, int N>
void BenchmarkMatrixMultiplication() {
    const int WARMUP_RUNS = 3;
    const int BENCH_RUNS = 10;
    
    auto* A = new SimpleMatrix<M, K>();
    auto* B = new SimpleMatrix<K, N>();
    auto* C = new SimpleMatrix<M, N>();
    
    FillRandom(A);
    FillRandom(B);
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        A->MatrixMultiplication(B, C);
    }
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < BENCH_RUNS; i++) {
        A->MatrixMultiplication(B, C);
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start).count();
    double avg_time_ms = duration / 1000.0 / BENCH_RUNS;
    
    double ops = 2.0 * M * N * K;
    double gflops = (ops * BENCH_RUNS) / (duration / 1e6) / 1e9;
    
    cout << "  " << setw(20) << left << (to_string(M) + "x" + to_string(K) + " @ " + to_string(K) + "x" + to_string(N))
         << setw(15) << fixed << setprecision(3) << avg_time_ms << " ms"
         << setw(15) << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
    
    delete A;
    delete B;
    delete C;
}

int main() {
    cout << "========================================" << endl;
    cout << "Matrix Multiplication Optimization Test" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    cout << "Correctness Tests:" << endl;
    cout << "==================" << endl;
    
    bool all_passed = true;
    
    all_passed &= TestMatrixMultiplication<4, 4, 4>();
    all_passed &= TestMatrixMultiplication<8, 8, 8>();
    all_passed &= TestMatrixMultiplication<16, 16, 16>();
    all_passed &= TestMatrixMultiplication<32, 32, 32>();
    all_passed &= TestMatrixMultiplication<64, 64, 64>();
    all_passed &= TestMatrixMultiplication<128, 128, 128>();
    all_passed &= TestMatrixMultiplication<256, 256, 256>();
    all_passed &= TestMatrixMultiplication<10, 784, 128>();
    all_passed &= TestMatrixMultiplication<128, 784, 10>();
    
    cout << endl;
    
    if (all_passed) {
        cout << "✅ All correctness tests passed!" << endl;
    } else {
        cout << "❌ Some tests failed!" << endl;
        return 1;
    }
    
    cout << endl;
    cout << "Performance Benchmarks:" << endl;
    cout << "=======================" << endl;
    cout << "  " << setw(20) << left << "Size" 
         << setw(15) << "Time" 
         << setw(15) << "Performance" << endl;
    cout << "  " << string(48, '-') << endl;
    
    BenchmarkMatrixMultiplication<64, 64, 64>();
    BenchmarkMatrixMultiplication<128, 128, 128>();
    BenchmarkMatrixMultiplication<256, 256, 256>();
    BenchmarkMatrixMultiplication<512, 512, 512>();
    BenchmarkMatrixMultiplication<10, 784, 128>();
    BenchmarkMatrixMultiplication<128, 784, 10>();
    
    cout << endl;
    cout << "========================================" << endl;
    cout << "✅ Testing complete!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}

#!/usr/bin/env python3
"""
Benchmark script for matrix multiplication performance testing.
Compares optimized matrix multiplication against numpy baseline.
"""

import time
import numpy as np
from Neurocore.network.Matrix import Matrix, PreCompileMatrix
from Neurocore.network.Config import Config

def benchmark_matrix_multiplication(sizes):
    """
    Benchmark matrix multiplication for various matrix sizes.
    
    Args:
        sizes: List of (M, N, K) tuples where M x K @ K x N
    """
    Config.VERBOSE = False
    
    print("=" * 80)
    print("Matrix Multiplication Performance Benchmark")
    print("=" * 80)
    print(f"{'Size (MxK @ KxN)':<25} {'Neurocore (ms)':<20} {'NumPy (ms)':<20} {'Speedup':<15}")
    print("-" * 80)
    
    results = []
    
    for M, K, N in sizes:
        # Pre-compile matrix types
        PreCompileMatrix(M, K, 1)
        PreCompileMatrix(K, N, 1)
        PreCompileMatrix(M, N, 1)
        
        # Generate random matrices
        np_a = np.random.randn(M, K).astype(np.float32)
        np_b = np.random.randn(K, N).astype(np.float32)
        
        # Warmup runs
        for _ in range(2):
            _ = np_a @ np_b
        
        # Benchmark NumPy
        numpy_times = []
        for _ in range(5):
            start = time.perf_counter()
            np_result = np_a @ np_b
            numpy_times.append((time.perf_counter() - start) * 1000)
        numpy_time = np.median(numpy_times)
        
        # Create Neurocore matrices
        mat_a = Matrix(M, K, 1, numpyArray=np_a)
        mat_b = Matrix(K, N, 1, numpyArray=np_b)
        mat_c = Matrix(M, N, 1)
        
        # Warmup Neurocore
        for _ in range(2):
            mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
        
        # Benchmark Neurocore
        neuro_times = []
        for _ in range(5):
            start = time.perf_counter()
            mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
            neuro_times.append((time.perf_counter() - start) * 1000)
        neuro_time = np.median(neuro_times)
        
        # Verify correctness
        neuro_result = mat_c.cpp_mat.to_numpy()
        max_diff = np.max(np.abs(np_result - neuro_result))
        
        speedup = numpy_time / neuro_time
        status = "✓" if max_diff < 1e-3 else "✗"
        
        size_str = f"{M}x{K} @ {K}x{N}"
        print(f"{size_str:<25} {neuro_time:>18.3f}  {numpy_time:>18.3f}  {speedup:>13.2f}x {status}")
        
        results.append({
            'size': (M, K, N),
            'neuro_time': neuro_time,
            'numpy_time': numpy_time,
            'speedup': speedup,
            'max_diff': max_diff
        })
    
    print("-" * 80)
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print("=" * 80)
    
    return results

def benchmark_transpose_operations(sizes):
    """
    Benchmark transpose-based matrix operations.
    """
    Config.VERBOSE = False
    
    print("\n" + "=" * 80)
    print("Transpose Operations Performance Benchmark")
    print("=" * 80)
    print(f"{'Operation':<30} {'Size':<20} {'Time (ms)':<20}")
    print("-" * 80)
    
    for M, K in sizes:
        # Pre-compile
        PreCompileMatrix(M, K, 1)
        PreCompileMatrix(K, M, 1)
        PreCompileMatrix(M, M, 1)
        
        # Generate matrices
        np_a = np.random.randn(M, K).astype(np.float32)
        np_b = np.random.randn(M, K).astype(np.float32)
        
        mat_a = Matrix(M, K, 1, numpyArray=np_a)
        mat_b = Matrix(M, K, 1, numpyArray=np_b)
        mat_c = Matrix(M, M, 1)
        
        # Benchmark A * B^T
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mat_a.cpp_mat.cross_product_with_transpose(mat_b.cpp_mat, mat_c.cpp_mat)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.median(times)
        print(f"{'A * B^T':<30} {f'{M}x{K}':<20} {avg_time:>18.3f}")
        
        # Benchmark A^T * B
        PreCompileMatrix(K, K, 1)
        mat_d = Matrix(K, K, 1)
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            mat_a.cpp_mat.cross_product_with_self_transpose(mat_b.cpp_mat, mat_d.cpp_mat)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.median(times)
        print(f"{'A^T * B':<30} {f'{M}x{K}':<20} {avg_time:>18.3f}")
    
    print("=" * 80)

if __name__ == "__main__":
    print("\nStarting Matrix Multiplication Benchmarks...\n")
    
    # Test various matrix sizes commonly used in neural networks
    test_sizes = [
        # Small matrices
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        
        # Rectangular matrices (common in FCL layers)
        (128, 784, 10),
        (256, 128, 64),
        (1000, 784, 128),
        
        # Large matrices
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    try:
        results = benchmark_matrix_multiplication(test_sizes)
        
        # Transpose operations
        transpose_sizes = [
            (128, 784),
            (256, 128),
            (512, 512),
        ]
        benchmark_transpose_operations(transpose_sizes)
        
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()

    print("\nBenchmark completed!")

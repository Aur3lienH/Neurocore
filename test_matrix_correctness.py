#!/usr/bin/env python3
"""
Test script to verify correctness of optimized matrix multiplication.
Tests against NumPy's implementation.
"""

import numpy as np
from Neurocore.network.Matrix import Matrix, PreCompileMatrix
from Neurocore.network.Config import Config

def test_matrix_multiplication():
    """Test basic matrix multiplication correctness."""
    Config.VERBOSE = False
    
    print("Testing Matrix Multiplication Correctness...")
    
    # Test 1: Small square matrices
    print("  Test 1: 4x4 @ 4x4...", end=" ")
    M, K, N = 4, 4, 4
    PreCompileMatrix(M, K, 1)
    PreCompileMatrix(K, N, 1)
    PreCompileMatrix(M, N, 1)
    
    np_a = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]], dtype=np.float32)
    np_b = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    mat_b = Matrix(K, N, 1, numpyArray=np_b)
    mat_c = Matrix(M, N, 1)
    
    mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
    neuro_result = mat_c.cpp_mat.to_numpy()
    np_result = np_a @ np_b
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-5, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    # Test 2: Non-square matrices
    print("  Test 2: 3x5 @ 5x7...", end=" ")
    M, K, N = 3, 5, 7
    PreCompileMatrix(M, K, 1)
    PreCompileMatrix(K, N, 1)
    PreCompileMatrix(M, N, 1)
    
    np_a = np.random.randn(M, K).astype(np.float32)
    np_b = np.random.randn(K, N).astype(np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    mat_b = Matrix(K, N, 1, numpyArray=np_b)
    mat_c = Matrix(M, N, 1)
    
    mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
    neuro_result = mat_c.cpp_mat.to_numpy()
    np_result = np_a @ np_b
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-4, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    # Test 3: Larger matrices
    print("  Test 3: 128x64 @ 64x32...", end=" ")
    M, K, N = 128, 64, 32
    PreCompileMatrix(M, K, 1)
    PreCompileMatrix(K, N, 1)
    PreCompileMatrix(M, N, 1)
    
    np_a = np.random.randn(M, K).astype(np.float32)
    np_b = np.random.randn(K, N).astype(np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    mat_b = Matrix(K, N, 1, numpyArray=np_b)
    mat_c = Matrix(M, N, 1)
    
    mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
    neuro_result = mat_c.cpp_mat.to_numpy()
    np_result = np_a @ np_b
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-3, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    # Test 4: Very large matrices
    print("  Test 4: 256x256 @ 256x256...", end=" ")
    M, K, N = 256, 256, 256
    PreCompileMatrix(M, K, 1)
    PreCompileMatrix(K, N, 1)
    PreCompileMatrix(M, N, 1)
    
    np_a = np.random.randn(M, K).astype(np.float32)
    np_b = np.random.randn(K, N).astype(np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    mat_b = Matrix(K, N, 1, numpyArray=np_b)
    mat_c = Matrix(M, N, 1)
    
    mat_a.cpp_mat.matrix_multiplication(mat_b.cpp_mat, mat_c.cpp_mat)
    neuro_result = mat_c.cpp_mat.to_numpy()
    np_result = np_a @ np_b
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-2, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    print("\n✅ All matrix multiplication tests passed!")

def test_transpose_operations():
    """Test transpose-based operations."""
    Config.VERBOSE = False
    
    print("\nTesting Transpose Operations...")
    
    # Test CrossProductWithTranspose (A * B^T)
    print("  Test 1: A * B^T (64x32, 64x32)...", end=" ")
    M, K = 64, 32
    PreCompileMatrix(M, K, 1)
    PreCompileMatrix(M, M, 1)
    
    np_a = np.random.randn(M, K).astype(np.float32)
    np_b = np.random.randn(M, K).astype(np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    mat_b = Matrix(M, K, 1, numpyArray=np_b)
    mat_c = Matrix(M, M, 1)
    
    mat_a.cpp_mat.cross_product_with_transpose(mat_b.cpp_mat, mat_c.cpp_mat)
    neuro_result = mat_c.cpp_mat.to_numpy()
    np_result = np_a @ np_b.T
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-3, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    # Test CrossProductWithSelfTranspose (A^T * B)
    print("  Test 2: A^T * B (64x32, 64x32)...", end=" ")
    PreCompileMatrix(K, K, 1)
    mat_d = Matrix(K, K, 1)
    
    mat_a.cpp_mat.cross_product_with_self_transpose(mat_b.cpp_mat, mat_d.cpp_mat)
    neuro_result = mat_d.cpp_mat.to_numpy()
    np_result = np_a.T @ np_b
    
    max_diff = np.max(np.abs(np_result - neuro_result))
    assert max_diff < 1e-3, f"Max difference: {max_diff}"
    print(f"✓ (max diff: {max_diff:.2e})")
    
    print("\n✅ All transpose operation tests passed!")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Matrix Multiplication Correctness Tests")
        print("=" * 60)
        print()
        
        test_matrix_multiplication()
        test_transpose_operations()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

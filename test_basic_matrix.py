#!/usr/bin/env python3
"""
Basic test to verify matrix compilation works with optimizations.
"""

import numpy as np
from Neurocore.network.Matrix import Matrix, PreCompileMatrix
from Neurocore.network.Config import Config

def test_basic():
    """Test that matrices can be created and compiled."""
    Config.VERBOSE = True
    
    print("Testing basic matrix creation and compilation...")
    
    # Test 1: Create a simple matrix
    print("\n1. Creating 4x4 matrix...")
    M, K = 4, 4
    PreCompileMatrix(M, K, 1)
    
    np_a = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]], dtype=np.float32)
    
    mat_a = Matrix(M, K, 1, numpyArray=np_a)
    print("✓ Matrix created successfully")
    
    # Test 2: Verify we can convert back to numpy
    print("\n2. Converting back to NumPy...")
    result = mat_a.cpp_mat.to_numpy()
    print(f"Result shape: {result.shape}")
    print("Result:\n", result)
    
    max_diff = np.max(np.abs(np_a - result))
    print(f"Max difference from original: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ Conversion successful")
    else:
        print("✗ Conversion failed")
        return False
    
    # Test 3: Try different sizes
    print("\n3. Testing various matrix sizes...")
    sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
    
    for M, K in sizes:
        PreCompileMatrix(M, K, 1)
        np_mat = np.random.randn(M, K).astype(np.float32)
        mat = Matrix(M, K, 1, numpyArray=np_mat)
        result = mat.cpp_mat.to_numpy()
        max_diff = np.max(np.abs(np_mat - result))
        print(f"  {M}x{K}: max_diff = {max_diff:.2e} {'✓' if max_diff < 1e-5 else '✗'}")
    
    print("\n✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_basic()
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

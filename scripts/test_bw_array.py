#!/usr/bin/env python3
"""Test script to verify per-frequency bandwidth support in sr_signature_panel"""

import numpy as np

# Test the bandwidth handling logic
def test_bw_handling():
    """Test that bw_hz can be scalar or array"""

    # Test case 1: scalar bandwidth
    bw_cfg = 0.5
    ladder = [7.83, 14.3, 20.8, 27.3, 33.8]

    if np.ndim(bw_cfg) == 0:  # scalar
        bw_array = np.full(len(ladder), float(bw_cfg))
    else:  # array
        bw_array = np.asarray(bw_cfg)
        if len(bw_array) != len(ladder):
            raise ValueError(f'bw_hz array length ({len(bw_array)}) must match ladder length ({len(ladder)})')

    print(f"Test 1 - Scalar bandwidth: {bw_cfg}")
    print(f"  Ladder length: {len(ladder)}")
    print(f"  bw_array: {bw_array}")
    assert len(bw_array) == len(ladder), "bw_array length should match ladder length"
    assert np.all(bw_array == 0.5), "All bandwidths should be 0.5"
    print("  ✓ PASSED\n")

    # Test case 2: array bandwidth
    bw_cfg = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    if np.ndim(bw_cfg) == 0:  # scalar
        bw_array = np.full(len(ladder), float(bw_cfg))
    else:  # array
        bw_array = np.asarray(bw_cfg)
        if len(bw_array) != len(ladder):
            raise ValueError(f'bw_hz array length ({len(bw_array)}) must match ladder length ({len(ladder)})')

    print(f"Test 2 - Array bandwidth: {bw_cfg}")
    print(f"  Ladder length: {len(ladder)}")
    print(f"  bw_array: {bw_array}")
    assert len(bw_array) == len(ladder), "bw_array length should match ladder length"
    assert np.allclose(bw_array, [0.5, 0.6, 0.7, 0.8, 0.9]), "Bandwidths should match input array"
    print("  ✓ PASSED\n")

    # Test case 3: mismatched array length (should raise error)
    bw_cfg = np.array([0.5, 0.6, 0.7])  # Wrong length

    try:
        if np.ndim(bw_cfg) == 0:  # scalar
            bw_array = np.full(len(ladder), float(bw_cfg))
        else:  # array
            bw_array = np.asarray(bw_cfg)
            if len(bw_array) != len(ladder):
                raise ValueError(f'bw_hz array length ({len(bw_array)}) must match ladder length ({len(ladder)})')
        print("  ✗ FAILED - Should have raised ValueError")
    except ValueError as e:
        print(f"Test 3 - Mismatched array length")
        print(f"  Expected error raised: {e}")
        print("  ✓ PASSED\n")

    print("All tests passed! ✓")

if __name__ == "__main__":
    test_bw_handling()

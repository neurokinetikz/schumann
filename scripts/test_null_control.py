#!/usr/bin/env python3
"""Quick test of null control functions with a single file"""

import os
from null_control_1_random_triplets import (
    detect_sie_events, generate_random_triplets,
    calculate_phi_ratios, get_all_files
)

# Get first available file
files = get_all_files()
if len(files) == 0:
    print("No files found!")
    exit(1)

test_file = files[0]
print(f"Testing with: {test_file}")
print()

# Test SIE detection
print("1. Testing SIE detection...")
try:
    sie_triplets = detect_sie_events(test_file)
    print(f"   ✓ SIE detection successful: {len(sie_triplets)} event(s) found")
    if len(sie_triplets) > 0:
        t = sie_triplets[0]
        print(f"   Sample SIE: f1={t.f1:.2f} Hz, f2={t.f2:.2f} Hz, f3={t.f3:.2f} Hz")
except Exception as e:
    print(f"   ✗ SIE detection failed: {e}")
print()

# Test random triplet generation
print("2. Testing random triplet generation...")
try:
    random_triplets = generate_random_triplets(test_file, n_triplets=100)
    print(f"   ✓ Random generation successful: {len(random_triplets)} triplets")
    if len(random_triplets) > 0:
        t = random_triplets[0]
        print(f"   Sample random: f1={t.f1:.2f} Hz, f2={t.f2:.2f} Hz, f3={t.f3:.2f} Hz")
except Exception as e:
    print(f"   ✗ Random generation failed: {e}")
print()

# Test φ-ratio calculation
print("3. Testing φ-ratio calculation...")
try:
    if len(random_triplets) > 0:
        t = random_triplets[0]
        analysis = calculate_phi_ratios(t.f1, t.f2, t.f3, t)
        print(f"   ✓ φ-ratio calculation successful")
        print(f"   f2/f1 = {analysis.ratio_f2_f1:.3f} (expected φ² = 2.618)")
        print(f"   f3/f1 = {analysis.ratio_f3_f1:.3f} (expected φ³ = 4.236)")
        print(f"   f3/f2 = {analysis.ratio_f3_f2:.3f} (expected φ = 1.618)")
        print(f"   Mean φ-error: {analysis.mean_phi_error:.4f}")
except Exception as e:
    print(f"   ✗ φ-ratio calculation failed: {e}")
print()

print("All tests completed!")

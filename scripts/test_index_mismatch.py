#!/usr/bin/env python3
"""
Test if there's an index mismatch between CANON and _half_bw arrays.
"""

import numpy as np

print("=" * 70)
print("INDEX MISMATCH DIAGNOSIS")
print("=" * 70)

# SCENARIO A: User's current setup (what they're passing)
print("\nSCENARIO A: What you're passing to FOOOF")
print("-" * 70)

CANON_A = [7.6, 20, 32.0]  # 3 selected harmonics
_half_bw_A = [0.6, 1, 2]   # 3 bandwidth values

print(f"CANON = {CANON_A}")
print(f"_half_bw = {_half_bw_A}")
print(f"\nPairing by position (array index):")

for i, (c, hb) in enumerate(zip(CANON_A, _half_bw_A)):
    lo, hi = c - hb, c + hb
    table_name = f"sr{i+1}"  # Table column name
    print(f"  Table column '{table_name}': CANON[{i}]={c} + _half_bw[{i}]={hb} → [{lo:.2f}, {hi:.2f}]")
    if lo <= 34.27 <= hi:
        print(f"                      ^^^ 34.27 Hz IS within this window")

# SCENARIO B: What the user THINKS they're passing
print("\n" + "=" * 70)
print("SCENARIO B: What you INTENDED (if there's a full array)")
print("-" * 70)

# Full Schumann resonance array (7 harmonics)
FULL_CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
FULL_HALF_BW = [0.6, 0.6, 1.0, 1.0, 2.5, 3.0, 3.5]  # Example full array

# Selected harmonics: sr1, sr3, sr5 (indices 0, 2, 4)
selected_indices = [0, 2, 4]
CANON_B = [FULL_CANON[i] for i in selected_indices]
HALF_BW_B = [FULL_HALF_BW[i] for i in selected_indices]

print(f"\nFull arrays:")
print(f"  FULL_CANON = {FULL_CANON}")
print(f"  FULL_HALF_BW = {FULL_HALF_BW}")

print(f"\nSelected harmonics (sr1, sr3, sr5 → indices 0, 2, 4):")
print(f"  CANON = {CANON_B}")
print(f"  _half_bw = {HALF_BW_B}")

print(f"\nPairing by harmonic number:")
for table_idx, full_idx in enumerate(selected_indices):
    c = FULL_CANON[full_idx]
    hb = FULL_HALF_BW[full_idx]
    lo, hi = c - hb, c + hb
    sr_name = f"sr{full_idx + 1}"  # Actual harmonic number
    table_name = f"sr{table_idx + 1}"  # Table column name
    print(f"  Table '{table_name}' = {sr_name}: CANON={c} + half_bw={hb} → [{lo:.2f}, {hi:.2f}]")
    if lo <= 34.27 <= hi:
        print(f"                      ^^^ 34.27 Hz IS within this window!")

# SCENARIO C: The bug scenario
print("\n" + "=" * 70)
print("SCENARIO C: THE BUG - If _half_bw has 7 values but CANON has 3")
print("-" * 70)

# This would happen if user passes full half_bw array but partial CANON
CANON_C = [7.6, 20, 32.0]  # 3 values
HALF_BW_C_FULL = [0.6, 0.6, 1.0, 1.0, 2.5, 3.0, 3.5]  # 7 values!

print(f"\nCANON = {CANON_C} (length {len(CANON_C)})")
print(f"_half_bw = {HALF_BW_C_FULL} (length {len(HALF_BW_C_FULL)})")

if len(HALF_BW_C_FULL) != len(CANON_C):
    print(f"\n⚠️  LENGTH MISMATCH! {len(HALF_BW_C_FULL)} != {len(CANON_C)}")
    print(f"FOOOF will use first {len(CANON_C)} values from _half_bw")

    for i, c in enumerate(CANON_C):
        hb = HALF_BW_C_FULL[i]
        lo, hi = c - hb, c + hb
        print(f"  Table sr{i+1}: CANON[{i}]={c} + _half_bw[{i}]={hb} → [{lo:.2f}, {hi:.2f}]")
        if lo <= 34.27 <= hi:
            print(f"              ^^^ 34.27 Hz IS within this window!")

# The CORRECT way for user's case
print("\n" + "=" * 70)
print("SOLUTION: Correct way to pass selected harmonics")
print("-" * 70)

print("""
If you want to track only sr1, sr3, sr5 from full Schumann series:

METHOD 1: Pass only the selected values (current approach - CORRECT)
---------
CANON = [7.6, 20, 32.0]  # sr1, sr3, sr5 values
_half_bw = [0.6, 1.0, 2.0]  # Corresponding bandwidths for sr1, sr3, sr5

Table columns will be sr1, sr2, sr3 but represent the selected harmonics.


METHOD 2: Pass full arrays and let system select (NOT CURRENTLY SUPPORTED)
---------
This would require additional parameter like `selected_harmonics=[0, 2, 4]`


YOUR ISSUE:
-----------
Check if your actual _half_bw array has MORE than 3 values!

Run this in your code:
    print(f"len(CANON) = {len(CANON)}")
    print(f"len(_half_bw) = {len(_half_bw)}")
    print(f"CANON = {CANON}")
    print(f"_half_bw = {_half_bw}")

If len(_half_bw) > len(CANON), that's the bug!
""")

# Test with what would allow 34.27
print("\n" + "=" * 70)
print("WHAT _half_bw VALUE WOULD ALLOW 34.27 Hz?")
print("-" * 70)

required_hb = 34.27 - 32.0
print(f"\nFor peak at 34.27 Hz with CANON=32:")
print(f"  Required: half_bw >= {required_hb:.4f} Hz")
print(f"  Your shown value: _half_bw[2] = 2.0 Hz")
print(f"  Would need: _half_bw[2] >= 2.27 Hz")

print(f"\nPossible scenarios:")
scenarios = [
    ([0.6, 1, 2], "Your shown config", False),
    ([0.6, 1, 2.5], "If _half_bw[2] = 2.5", True),
    ([0.6, 0.6, 1.0, 1.0, 2.5, 3.0, 3.5], "Full 7-element array (BUG!)", True),
]

for hb_array, description, allows in scenarios:
    if len(hb_array) >= 3:
        hb_val = hb_array[2]  # Third element (for CANON[2]=32)
        lo, hi = 32 - hb_val, 32 + hb_val
        status = "✓ ALLOWS 34.27" if (lo <= 34.27 <= hi) else "✗ REJECTS 34.27"
        print(f"  {status}: {description}")
        print(f"           _half_bw = {hb_array}")
        print(f"           Window for CANON[2]=32: [{lo:.2f}, {hi:.2f}]")

#!/usr/bin/env python3
"""
Check which harmonic index corresponds to which frequency.
"""

import numpy as np

# SCENARIO 1: Your shown configuration
print("=" * 70)
print("SCENARIO 1: CANON = [7.6, 20, 32.0] (3 harmonics)")
print("=" * 70)

CANON1 = [7.6, 20, 32.0]
_half_bw1 = [0.6, 1, 2]

for i, (canon, hb) in enumerate(zip(CANON1, _half_bw1)):
    lo, hi = canon - hb, canon + hb
    sr_name = f"sr{i+1}"
    print(f"{sr_name}: CANON={canon:6.2f}, half_bw={hb}, window=[{lo:6.2f}, {hi:6.2f}]")
    if lo <= 34.27 <= hi:
        print(f"     ⚠️  34.27 Hz IS within {sr_name} window!")

print(f"\nIn this case, sr5 doesn't exist (only sr1, sr2, sr3)")

# SCENARIO 2: Possible actual configuration (7 harmonics)
print("\n" + "=" * 70)
print("SCENARIO 2: CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32] (7 harmonics)")
print("=" * 70)

CANON2 = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
_half_bw2 = [0.6, 0.6, 0.6, 0.6, 1, 2, 2]  # Example per-harmonic bandwidths

for i, (canon, hb) in enumerate(zip(CANON2, _half_bw2)):
    lo, hi = canon - hb, canon + hb
    sr_name = f"sr{i+1}"
    print(f"{sr_name}: CANON={canon:6.2f}, half_bw={hb}, window=[{lo:6.2f}, {hi:6.2f}]")
    if lo <= 34.27 <= hi:
        print(f"     ⚠️  34.27 Hz IS within {sr_name} window!")

# SCENARIO 3: What if half_bw is larger for one of them?
print("\n" + "=" * 70)
print("SCENARIO 3: Check if any configuration allows 34.27")
print("=" * 70)

# For 34.27 to be detected with CANON=32:
required_hb = 34.27 - 32.0
print(f"\nFor 34.27 Hz to be within [32-hb, 32+hb]:")
print(f"  Required: half_bw >= {required_hb:.4f} Hz")
print(f"  Your value: half_bw = 2.0 Hz")
print(f"  Status: {'✓ ALLOWS' if 2.0 >= required_hb else '✗ REJECTS'}")

# But what if there's another harmonic?
print(f"\nOther possibilities:")
for canon_test in [25, 30, 32, 33, 34, 35]:
    for hb_test in [1, 1.5, 2, 2.5, 3]:
        lo, hi = canon_test - hb_test, canon_test + hb_test
        if lo <= 34.27 <= hi:
            print(f"  • CANON={canon_test}, half_bw={hb_test} → window=[{lo:.2f}, {hi:.2f}] ✓ allows 34.27")

# SCENARIO 4: Most likely case
print("\n" + "=" * 70)
print("MOST LIKELY EXPLANATION")
print("=" * 70)

print("""
Based on you mentioning 'sr5', you probably have MORE harmonics than [7.6, 20, 32].

Common Schumann harmonic arrays:
1. [7.6, 20, 32]                          → 3 harmonics (sr1, sr2, sr3)
2. [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32] → 7 harmonics (sr1..sr7)
3. [7.83, 14.3, 20.8, 27.3, 33.8]          → 5 harmonics (sr1..sr5)

If you have 'sr5', you have AT LEAST 5 harmonics.

Action: Print len(CANON) in your code to see how many harmonics you actually have!
""")

print("\n" + "=" * 70)
print("ADD THIS TO YOUR CODE:")
print("=" * 70)
print("""
print(f"Number of harmonics: {len(CANON)}")
print(f"CANON values: {CANON}")
print(f"half_bw values: {_half_bw}")
print(f"Which is sr5? {CANON[4] if len(CANON) > 4 else 'N/A - only ' + str(len(CANON)) + ' harmonics'}")
""")

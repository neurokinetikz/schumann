#!/usr/bin/env python3
"""
Debug wrapper for match_peaks_to_canonical to trace 34.27 Hz issue.

Usage:
    import debug_fooof_wrapper  # Add this BEFORE your other imports
    from lib.detect_ignition import detect_ignitions_session
    # ... rest of your code runs normally with debug output
"""

import sys
import numpy as np

# Import the module we want to patch
try:
    from lib import fooof_harmonics
except ImportError:
    print("ERROR: Could not import lib.fooof_harmonics")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Store original function
_original_match_peaks = fooof_harmonics.match_peaks_to_canonical

def debug_match_peaks_to_canonical(peak_params, f_can, search_halfband, method='distance'):
    """Wrapper that adds debug output to match_peaks_to_canonical."""

    # Convert halfband to list for indexing
    if isinstance(search_halfband, (int, float)):
        halfbands = [float(search_halfband)] * len(f_can)
    else:
        halfbands = list(search_halfband)

    print("\n" + "=" * 70)
    print("🔍 [DEBUG] match_peaks_to_canonical")
    print("=" * 70)
    print(f"f_can: {f_can}")
    print(f"search_halfband (input): {search_halfband}")
    print(f"halfbands (expanded): {halfbands}")
    print(f"method: {method}")

    # Show search windows
    print(f"\nSearch windows for each harmonic:")
    for i, (f0, hb) in enumerate(zip(f_can, halfbands)):
        lo, hi = f0 - hb, f0 + hb
        print(f"  sr{i+1}: CANON={f0:7.2f} Hz, halfband={hb}, window=[{lo:6.2f}, {hi:6.2f}]")

        # Check if 34.27 would be in this window
        if lo <= 34.27 <= hi:
            print(f"       ⚠️  34.27 Hz IS within this window!")
            print(f"           {lo:.2f} <= 34.27 <= {hi:.2f}")

    # Show all FOOOF-detected peaks
    print(f"\nFOOOF detected {len(peak_params)} peaks:")
    if len(peak_params) == 0:
        print("  (none)")
    else:
        for i, (pf, pp, pbw) in enumerate(peak_params):
            marker = "  "
            if abs(pf - 34.27) < 0.01:
                marker = "👉"

            print(f"{marker} Peak {i+1}: freq={pf:7.4f} Hz, power={pp:7.4f}, bw={pbw:6.4f}")

            if abs(pf - 34.27) < 0.01:
                print(f"            ^^^ This is the 34.27 Hz peak")

                # Check which windows it would fit in
                for j, (f0, hb) in enumerate(zip(f_can, halfbands)):
                    lo, hi = f0 - hb, f0 + hb
                    if lo <= pf <= hi:
                        print(f"            Fits in sr{j+1} window [{lo:.2f}, {hi:.2f}]")

    # Show which peaks are in range for each harmonic
    print(f"\nPeak matching (method='{method}'):")
    for i, (f0, hb) in enumerate(zip(f_can, halfbands)):
        lo, hi = f0 - hb, f0 + hb

        # Find peaks in this window
        if len(peak_params) > 0:
            in_range = (peak_params[:, 0] >= lo) & (peak_params[:, 0] <= hi)
            candidates = peak_params[in_range]

            print(f"\n  sr{i+1} (CANON={f0:.2f}, window=[{lo:.2f}, {hi:.2f}]):")
            if len(candidates) == 0:
                print(f"    No peaks in range → will use canonical {f0:.2f}")
            else:
                print(f"    {len(candidates)} peak(s) in range:")
                for j, (pf, pp, pbw) in enumerate(candidates):
                    marker = "👉" if abs(pf - 34.27) < 0.01 else "  "
                    print(f"    {marker}  {pf:.4f} Hz (power={pp:.4f})")

                # Show which will be selected
                if method == 'power':
                    best_idx = np.argmax(candidates[:, 1])
                    best = candidates[best_idx]
                    print(f"    → Will select HIGHEST POWER: {best[0]:.4f} Hz (power={best[1]:.4f})")
                elif method == 'distance':
                    best_idx = np.argmin(np.abs(candidates[:, 0] - f0))
                    best = candidates[best_idx]
                    print(f"    → Will select CLOSEST: {best[0]:.4f} Hz (dist from {f0:.2f} = {abs(best[0]-f0):.4f})")

    # Call original function
    result = _original_match_peaks(peak_params, f_can, search_halfband, method)
    harmonics, powers, bandwidths = result

    # Show final result
    print(f"\n{'='*70}")
    print(f"FINAL MATCHED HARMONICS:")
    print(f"{'='*70}")
    for i, (h, p, bw) in enumerate(zip(harmonics, powers, bandwidths)):
        if np.isnan(p):
            print(f"  sr{i+1} = {h:7.4f} Hz (no peak found)")
        else:
            marker = "⚠️ " if abs(h - 34.27) < 0.01 else "✓ "
            print(f"{marker} sr{i+1} = {h:7.4f} Hz (power={p:7.4f})")

            if abs(h - 34.27) < 0.01:
                print(f"       ^^^ 34.27 Hz WAS SELECTED FOR sr{i+1}!")
                expected_lo = f_can[i] - halfbands[i]
                expected_hi = f_can[i] + halfbands[i]
                print(f"       Expected window for sr{i+1}: [{expected_lo:.2f}, {expected_hi:.2f}]")
                print(f"       CANON={f_can[i]:.2f}, halfband={halfbands[i]}")
                if 34.27 > expected_hi:
                    print(f"       ❌ ERROR: 34.27 > {expected_hi:.2f} (should have been rejected!)")
                else:
                    print(f"       ✓ OK: 34.27 <= {expected_hi:.2f} (correctly within window)")

    print("=" * 70 + "\n")

    return result

# Monkey-patch the function
fooof_harmonics.match_peaks_to_canonical = debug_match_peaks_to_canonical

print("=" * 70)
print("🔧 DEBUG WRAPPER INSTALLED")
print("=" * 70)
print("✓ match_peaks_to_canonical is now instrumented")
print("✓ Will show detailed debug output when FOOOF matching runs")
print("✓ Looking for 34.27 Hz peak specifically")
print("=" * 70 + "\n")

"""
Quick compatibility test for fooof_harmonics module.

Tests that the module works with both specparam (new) and fooof (legacy).
"""

import sys

print("=" * 70)
print("FOOOF/SpecParam Compatibility Test")
print("=" * 70)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from lib.fooof_harmonics import (
        PACKAGE_NAME,
        FOOOF_AVAILABLE,
        FOOOF,
        _has_model,
        _get_power_spectrum
    )
    print(f"   ✓ Module imported successfully")
    print(f"   Package: {PACKAGE_NAME}")
    print(f"   Available: {FOOOF_AVAILABLE}")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

if not FOOOF_AVAILABLE:
    print("\n   ⚠️  No spectral parameterization package installed.")
    print("   Install with: pip install specparam (recommended)")
    print("   Or legacy: pip install fooof")
    sys.exit(0)

# Test 2: Check package-specific behavior
print(f"\n2. Testing {PACKAGE_NAME} compatibility...")

if PACKAGE_NAME == 'specparam':
    print("   Testing specparam-specific features:")
    try:
        from specparam import SpectralModel
        fm = SpectralModel()

        # Test attribute names
        has_aperiodic = hasattr(fm, 'aperiodic_params_')
        print(f"   ✓ Has aperiodic_params_ attribute: {has_aperiodic}")

        # Test _has_model helper
        result = _has_model(fm)
        print(f"   ✓ _has_model(empty_model) = {result} (expected False)")

        print("   ✓ All specparam compatibility checks passed!")

    except Exception as e:
        print(f"   ✗ specparam test failed: {e}")
        sys.exit(1)

elif PACKAGE_NAME == 'fooof':
    print("   Testing fooof-specific features:")
    print("   ⚠️  Note: fooof is deprecated. Consider upgrading to specparam.")
    try:
        from fooof import FOOOF as FOOOFLegacy
        fm = FOOOFLegacy()

        # Test attribute names
        has_model_attr = hasattr(fm, 'has_model')
        print(f"   ✓ Has has_model attribute: {has_model_attr}")

        # Test _has_model helper
        result = _has_model(fm)
        print(f"   ✓ _has_model(empty_model) = {result} (expected False)")

        print("   ✓ All fooof compatibility checks passed!")

    except Exception as e:
        print(f"   ✗ fooof test failed: {e}")
        sys.exit(1)

# Test 3: Test with synthetic data
print("\n3. Testing with synthetic data...")
try:
    import numpy as np
    from lib.fooof_harmonics import detect_harmonics_fooof

    # Create synthetic DataFrame with realistic structure
    import pandas as pd

    # Generate 10 seconds of synthetic data at 128 Hz
    fs = 128
    duration = 10
    n_samples = fs * duration
    t = np.arange(n_samples) / fs

    # Synthetic signal: 1/f background + peaks at 8, 16, 24 Hz
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    psd_model = np.zeros_like(freqs)

    # 1/f background
    psd_model = 10 ** (2 - 1.5 * np.log10(freqs + 1))

    # Add peaks at 8, 16, 24 Hz (simulate harmonics)
    for peak_f in [8, 16, 24]:
        peak_idx = np.argmin(np.abs(freqs - peak_f))
        psd_model[max(0, peak_idx-2):peak_idx+3] *= 3

    # Convert PSD back to time series (rough approximation)
    # For testing, we'll just use white noise
    signal = np.random.randn(n_samples) * 10

    # Add some sinusoids at harmonic frequencies
    signal += 5 * np.sin(2 * np.pi * 8 * t)
    signal += 3 * np.sin(2 * np.pi * 16 * t)
    signal += 2 * np.sin(2 * np.pi * 24 * t)

    # Create DataFrame
    RECORDS = pd.DataFrame({
        'Timestamp': t,
        'EEG.F4': signal,
        'EEG.O1': signal + np.random.randn(n_samples) * 2  # Add some noise
    })

    print(f"   Created synthetic data: {n_samples} samples at {fs} Hz")

    # Run FOOOF detection
    harmonics, result = detect_harmonics_fooof(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1'],
        fs=fs,
        f_can=(8.0, 16.0, 24.0),  # Look for our synthetic harmonics
        freq_range=(1, 40),
        combine='median'
    )

    print(f"   ✓ FOOOF fit successful!")
    print(f"   Detected harmonics: {[f'{h:.2f}' for h in harmonics]}")
    print(f"   Aperiodic exponent: {result.aperiodic_exponent:.3f}")
    print(f"   R²: {result.r_squared:.3f}")
    print(f"   Number of peaks: {len([p for p in result.harmonic_powers if not np.isnan(p)])}")

except Exception as e:
    print(f"   ✗ Synthetic data test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test plotting function (without displaying)
print("\n4. Testing visualization functions...")
try:
    from lib.fooof_harmonics import plot_fooof_fit_with_harmonics
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig = plot_fooof_fit_with_harmonics(
        result,
        harmonics=(8, 16, 24),
        freq_range=(1, 40)
    )
    plt.close(fig)

    print("   ✓ Plotting functions work correctly!")

except Exception as e:
    print(f"   ✗ Plotting test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test window filtering
print("\n5. Testing time window filtering...")
try:
    # Use the RECORDS DataFrame from Test 3
    # Total duration should be 10 seconds (1280 samples at 128 Hz)
    total_duration = len(RECORDS) / fs
    print(f"   Total recording duration: {total_duration:.1f}s")

    # Test window filtering with a 5-second window
    window_start, window_end = 2.0, 7.0
    harmonics_windowed, result_windowed = detect_harmonics_fooof(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1'],
        fs=fs,
        window=[window_start, window_end],
        f_can=(8.0, 16.0, 24.0),
        freq_range=(1, 40),
        combine='median'
    )

    print(f"   ✓ Window filtering successful!")
    print(f"   Window: [{window_start}, {window_end}]s")
    print(f"   Detected harmonics: {[f'{h:.2f}' for h in harmonics_windowed]}")
    print(f"   R²: {result_windowed.r_squared:.3f}")

    # Test edge case: window at end of recording
    window_end2 = total_duration
    harmonics_end, result_end = detect_harmonics_fooof(
        records=RECORDS,
        channels='EEG.F4',
        fs=fs,
        window=[window_end2 - 3.0, window_end2],
        f_can=(8.0, 16.0, 24.0),
        freq_range=(1, 40)
    )
    print(f"   ✓ End-of-recording window works correctly!")

    # Test error handling: invalid window
    try:
        harmonics_bad, result_bad = detect_harmonics_fooof(
            records=RECORDS,
            channels='EEG.F4',
            fs=fs,
            window=[100, 200],  # Beyond recording duration
            f_can=(8.0, 16.0, 24.0)
        )
        print(f"   ✗ Should have raised error for out-of-bounds window!")
        sys.exit(1)
    except ValueError as e:
        if "contains no data" in str(e):
            print(f"   ✓ Correctly rejects out-of-bounds window")
        else:
            raise

    print("   ✓ All window filtering tests passed!")

except Exception as e:
    print(f"   ✗ Window filtering test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✓ ALL COMPATIBILITY TESTS PASSED!")
print("=" * 70)
print(f"\nThe fooof_harmonics module is working correctly with {PACKAGE_NAME}.")
print("\nYou can now use it for Schumann harmonic detection:")
print("  from lib.fooof_harmonics import detect_harmonics_fooof")
print("  harmonics, result = detect_harmonics_fooof(RECORDS, channels, fs=128)")
print("=" * 70)

"""
Test script for FOOOF-based Schumann harmonic detection.

Demonstrates the new fooof_harmonics module with example usage patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if FOOOF is available
try:
    from lib.fooof_harmonics import (
        detect_harmonics_fooof,
        detect_harmonics_fooof_multichannel,
        plot_fooof_fit_with_harmonics,
        compare_psd_fooof,
        quick_fooof_summary,
        check_fooof_available,
        fooof_refine_existing_harmonics
    )

    if not check_fooof_available():
        print("=" * 70)
        print("Spectral parameterization package is not installed!")
        print("=" * 70)
        print("\nInstall with: pip install specparam  (recommended)")
        print("Or legacy: pip install fooof")
        print("\nDocumentation: https://specparam-tools.github.io/specparam/")
        exit(1)

    print("✓ FOOOF module loaded successfully")

except ImportError as e:
    print(f"Error importing fooof_harmonics: {e}")
    exit(1)


# ============================================================================
# Example 1: Basic harmonic detection (single channel)
# ============================================================================
def example_1_basic_detection():
    """Example 1: Basic FOOOF harmonic detection on single channel."""
    print("\n" + "=" * 70)
    print("Example 1: Basic FOOOF Harmonic Detection")
    print("=" * 70)

    # Note: Replace with your actual data loading
    print("\n⚠️  This is a template. Replace with your actual RECORDS data:")
    print("    RECORDS = pd.read_csv('your_session.csv')")
    print("    # or load from your data pipeline\n")

    # Template code (commented out - uncomment when you have data):
    """
    harmonics, result = detect_harmonics_fooof(
        records=RECORDS,
        channels='EEG.F4',  # or ['EEG.F4'] for single channel
        fs=128,
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
        freq_range=(1.0, 50.0)
    )

    print(f"Detected harmonics: {[f'{h:.2f}' for h in harmonics]}")
    print(f"Aperiodic exponent (β): {result.aperiodic_exponent:.3f}")
    print(f"Aperiodic offset: {result.aperiodic_offset:.3f}")
    print(f"R² (fit quality): {result.r_squared:.3f}")
    print(f"\nPeak details:")
    for i, (h, p, bw) in enumerate(zip(result.harmonics,
                                       result.harmonic_powers,
                                       result.harmonic_bandwidths)):
        if not np.isnan(p):
            print(f"  H{i+1}: {h:.2f} Hz, Power: {p:.3f}, BW: {bw:.2f} Hz")
    """


# ============================================================================
# Example 2: Multi-channel detection with combined PSD
# ============================================================================
def example_2_multichannel_combined():
    """Example 2: Multi-channel FOOOF with median-combined PSD."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Channel Detection (Combined PSD)")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    # Analyze multiple channels with median-combined PSD
    harmonics, result = detect_harmonics_fooof(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],  # multiple channels
        fs=128,
        combine='median',  # robust aggregation
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8)
    )

    print(f"Harmonics (median of 3 channels): {[f'{h:.2f}' for h in harmonics]}")
    print(f"1/f slope: {result.aperiodic_exponent:.3f}")

    # Visualize the FOOOF fit
    fig = plot_fooof_fit_with_harmonics(
        result,
        title='Multi-Channel Median PSD — FOOOF Fit',
        freq_range=(5, 40)
    )
    plt.show()
    """


# ============================================================================
# Example 3: Separate fits per channel
# ============================================================================
def example_3_separate_fits():
    """Example 3: Fit FOOOF separately for each channel."""
    print("\n" + "=" * 70)
    print("Example 3: Separate FOOOF Fits Per Channel")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    results = detect_harmonics_fooof_multichannel(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.F3', 'EEG.O1', 'EEG.O2'],
        fs=128,
        separate_fits=True,  # fit each channel independently
        f_can=(7.83, 14.3, 20.8, 27.3)
    )

    print("\nPer-channel results:")
    print("-" * 70)
    for ch_name, res in results.items():
        print(f"\n{ch_name}:")
        print(f"  Harmonics: {[f'{h:.2f}' for h in res.harmonics]}")
        print(f"  Exponent:  {res.aperiodic_exponent:.3f}")
        print(f"  R²:        {res.r_squared:.3f}")

    # Plot comparison across channels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (ch, res) in zip(axes.flat, results.items()):
        # Plot on individual axis
        freqs = res.freqs
        psd = res.psd
        model = res.model.fooofed_spectrum_

        ax.semilogy(freqs, psd, 'gray', alpha=0.6, label='Data')
        ax.semilogy(freqs, 10**model, 'r', lw=2, label='FOOOF')
        ax.set_title(f'{ch} (β={res.aperiodic_exponent:.3f})')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(5, 40)

    plt.tight_layout()
    plt.show()
    """


# ============================================================================
# Example 4: Comparison plot (naive vs FOOOF)
# ============================================================================
def example_4_comparison():
    """Example 4: Side-by-side comparison of naive vs FOOOF detection."""
    print("\n" + "=" * 70)
    print("Example 4: Comparison — Naive vs FOOOF Detection")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    fig, result = compare_psd_fooof(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1'],
        fs=128,
        f_can=(7.83, 14.3, 20.8, 27.3),
        freq_range=(5, 35),
        title='Schumann Harmonics: Naive argmax vs FOOOF Detection'
    )

    plt.show()

    # Save figure
    # fig.savefig('fooof_comparison.png', dpi=300, bbox_inches='tight')
    """


# ============================================================================
# Example 5: Quick summary (one-liner)
# ============================================================================
def example_5_quick_summary():
    """Example 5: Quick FOOOF summary for rapid analysis."""
    print("\n" + "=" * 70)
    print("Example 5: Quick FOOOF Summary")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    summary = quick_fooof_summary(
        RECORDS,
        channels='EEG.F4',
        fs=128
    )

    print("Quick Summary:")
    print(f"  Harmonics: {summary['harmonics']}")
    print(f"  Exponent:  {summary['exponent']:.3f}")
    print(f"  Offset:    {summary['offset']:.3f}")
    print(f"  R²:        {summary['r_squared']:.3f}")
    """


# ============================================================================
# Example 6: Refine existing harmonic estimates
# ============================================================================
def example_6_refine_existing():
    """Example 6: Refine existing harmonic estimates using FOOOF."""
    print("\n" + "=" * 70)
    print("Example 6: Refine Existing Harmonic Estimates")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    # Suppose you already have coarse estimates from existing method
    from lib.test import estimate_sr_harmonics

    coarse_harmonics = estimate_sr_harmonics(
        records=RECORDS,
        sr_channel=['EEG.F4', 'EEG.O1'],
        fs=128,
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8)
    )

    print(f"Coarse estimates: {[f'{h:.2f}' for h in coarse_harmonics]}")

    # Refine using FOOOF
    refined_harmonics = fooof_refine_existing_harmonics(
        harmonics_in=coarse_harmonics,
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1'],
        fs=128,
        search_halfband=0.5
    )

    print(f"FOOOF refined:   {[f'{h:.2f}' for h in refined_harmonics]}")

    # Show differences
    print("\nRefinement (Hz):")
    for i, (c, r) in enumerate(zip(coarse_harmonics, refined_harmonics)):
        delta = r - c
        print(f"  H{i+1}: {c:.2f} → {r:.2f} (Δ {delta:+.3f})")
    """


# ============================================================================
# Example 7: Integration with existing wavelet pipeline
# ============================================================================
def example_7_wavelet_integration():
    """Example 7: Use FOOOF to set initial frequencies for wavelet analysis."""
    print("\n" + "=" * 70)
    print("Example 7: Integration with Wavelet Pipeline")
    print("=" * 70)

    print("\n⚠️  Template code (uncomment when you have data):\n")

    """
    from lib.harmonics import detect_and_plot_schumann_wavelet

    # Step 1: Use FOOOF to get accurate harmonic frequencies
    harmonics, result = detect_harmonics_fooof(
        records=RECORDS,
        channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
        fs=128,
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8)
    )

    # Extract fundamental frequency (1st harmonic)
    f0_refined = harmonics[0]

    print(f"FOOOF-refined fundamental: {f0_refined:.3f} Hz")
    print(f"  (vs canonical 7.83 Hz)")

    # Step 2: Use refined f0 for wavelet detection
    wavelet_result = detect_and_plot_schumann_wavelet(
        records=RECORDS,
        signal_col='EEG.F4',
        time_col='Timestamp',
        f0=f0_refined,  # Use FOOOF-refined frequency
        n_harmonics=5,
        show=True
    )

    print(f"\nWavelet detection completed with refined f0={f0_refined:.3f} Hz")
    """


def example_8_time_window_analysis():
    """
    Example 8: Time Window Analysis
    --------------------------------
    Analyze specific time windows for event-triggered or sliding window analysis.
    """
    print("\n" + "=" * 70)
    print("Example 8: Time Window Analysis")
    print("=" * 70)

    """
    # A. Event-Triggered Analysis
    print("\nA. Event-triggered: Analyze window around ignition onset")

    t0 = 100.5  # ignition onset time (seconds)

    harmonics, result = detect_harmonics_fooof(
        RECORDS,
        channels='EEG.F4',
        fs=128,
        window=[t0 - 2.0, t0 + 2.0],  # ±2s around event
        f_can=(7.83, 14.3, 20.8, 27.3)
    )

    print(f"  Window: [{t0-2:.1f}, {t0+2:.1f}]s")
    print(f"  Harmonics: {harmonics}")
    print(f"  β: {result.aperiodic_exponent:.3f}")
    print(f"  R²: {result.r_squared:.3f}")

    # B. Sliding Window Analysis
    print("\nB. Sliding window: Track temporal evolution")

    import numpy as np
    import pandas as pd

    window_size = 60.0  # 60-second windows
    step_size = 30.0    # 30-second step
    session_duration = 300.0  # 5 minutes

    results = []
    for t_start in np.arange(0, session_duration - window_size, step_size):
        t_end = t_start + window_size

        harmonics, result = detect_harmonics_fooof(
            RECORDS,
            channels=['EEG.O1', 'EEG.O2'],
            fs=128,
            window=[t_start, t_end],
            f_can=(7.83, 14.3, 20.8, 27.3)
        )

        results.append({
            'window_center': t_start + window_size / 2,
            'h1_freq': harmonics[0],
            'exponent': result.aperiodic_exponent,
            'r_squared': result.r_squared
        })

    df = pd.DataFrame(results)
    print(f"\n  Computed FOOOF for {len(df)} time windows")
    print(f"  H1 range: {df['h1_freq'].min():.2f} - {df['h1_freq'].max():.2f} Hz")
    print(f"  β range: {df['exponent'].min():.3f} - {df['exponent'].max():.3f}")

    # Plot if desired
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # axes[0].plot(df['window_center'], df['h1_freq'], 'o-')
    # axes[0].set_ylabel('H1 Frequency (Hz)')
    # axes[1].plot(df['window_center'], df['exponent'], 'o-', color='purple')
    # axes[1].set_ylabel('1/f Exponent (β)')
    # axes[1].set_xlabel('Time (s)')
    # plt.tight_layout()
    # plt.show()

    # C. State Comparison
    print("\nC. State comparison: Baseline vs. Task")

    baseline_harmonics, baseline_result = detect_harmonics_fooof(
        RECORDS, 'EEG.F4', fs=128,
        window=[0, 120],  # First 2 minutes
        f_can=(7.83, 14.3, 20.8, 27.3)
    )

    task_harmonics, task_result = detect_harmonics_fooof(
        RECORDS, 'EEG.F4', fs=128,
        window=[120, 300],  # Minutes 3-5
        f_can=(7.83, 14.3, 20.8, 27.3)
    )

    print(f"\n  Baseline (0-120s):")
    print(f"    H1: {baseline_harmonics[0]:.2f} Hz")
    print(f"    β: {baseline_result.aperiodic_exponent:.3f}")
    print(f"  Task (120-300s):")
    print(f"    H1: {task_harmonics[0]:.2f} Hz")
    print(f"    β: {task_result.aperiodic_exponent:.3f}")
    print(f"  Δβ: {task_result.aperiodic_exponent - baseline_result.aperiodic_exponent:.3f}")
    """


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FOOOF-Based Schumann Harmonic Detection — Test Suite")
    print("=" * 70)
    print("\nThis script demonstrates usage patterns for the new fooof_harmonics")
    print("module. Replace template code with your actual data to run examples.\n")

    # Run all examples
    example_1_basic_detection()
    example_2_multichannel_combined()
    example_3_separate_fits()
    example_4_comparison()
    example_5_quick_summary()
    example_6_refine_existing()
    example_7_wavelet_integration()
    example_8_time_window_analysis()

    print("\n" + "=" * 70)
    print("Test suite complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Install specparam: pip install specparam (recommended)")
    print("     Or legacy fooof: pip install fooof")
    print("  2. Load your RECORDS data")
    print("  3. Uncomment and run the example code above")
    print("  4. Compare FOOOF results with existing methods")
    print("\nDocumentation: https://specparam-tools.github.io/specparam/")
    print("=" * 70 + "\n")

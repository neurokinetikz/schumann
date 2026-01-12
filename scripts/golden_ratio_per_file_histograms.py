"""Per-File FOOOF Peak Frequency Histograms

Creates individual histograms of FOOOF-detected peaks for each EEG file.
"""

import sys; sys.path.insert(0, './lib')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utilities
from fooof_harmonics import detect_harmonics_fooof

# Constants
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034
FS = 128
ELECTRODES = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1',
              'EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']

# Schumann Resonance harmonics for reference
SR_HARMONICS = [7.83, 14.3, 20.8, 27.3, 33.8, 40.3]

def list_csv_files(directory):
    """List all CSV files in a directory."""
    if not os.path.exists(directory):
        return []
    return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

def list_mpeng_files(directory):
    """List all CSV files from MPENG directories (non-EEG files moved to other/)."""
    if not os.path.exists(directory):
        return []
    return [(directory + "/" + f, 0) for f in os.listdir(directory) if f.endswith('.csv')]

# Original FILES with header=1
FILES = [
    ('data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv', 1),
    ('data/Test_06.11.20_14.28.18.md.pm.bp.csv', 1),
    ('data/20201229_29.12.20_11.27.57.md.pm.bp.csv', 1),
    ('data/med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv', 1),
    ('data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv', 1),
]

# Other file sets
PHYSF = [(f, 0) for f in list_csv_files("data/PhySF")]
VEP = [(f, 0) for f in list_csv_files("data/vep")]
MPENG2 = list_mpeng_files("data/mpeng2")
MPENG1 = [(f, 0) for f in list_csv_files("data/mpeng1")]
ArEEG = [(f, 1) for f in list_csv_files("data/ArEEG")]

# SELECT WHICH FILES TO PROCESS
ALL_FILES = PHYSF  # Change this to process different file sets

# Output directory
OUTPUT_DIR = "fooof_histograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_peaks_from_file(filepath, header=1):
    """Extract all FOOOF peaks from a single EEG file."""
    try:
        RECORDS = utilities.load_eeg_csv(filepath, electrodes=ELECTRODES, fs=FS, header=header)
    except Exception as e:
        print(f"  Error loading: {e}")
        return None, None

    duration = len(RECORDS) / FS
    all_peaks = []

    for ch in ELECTRODES:
        try:
            harmonics, result = detect_harmonics_fooof(
                records=RECORDS,
                channels=[ch],
                fs=FS,
                freq_range=(1.0, 50.0),
                max_n_peaks=20,
                peak_threshold=0.001,
                min_peak_height=0.0001,
                peak_width_limits=(0.2, 20.0),
            )
            for p in result.all_peaks:
                p['channel'] = ch
                all_peaks.append(p)
        except Exception as e:
            pass

    return all_peaks, duration


def create_histogram(peak_freqs, filename, duration, output_path):
    """Create and save a histogram for a single file's peaks."""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram with fine bins (matching aggregate style)
    bins = np.arange(1, 51, 0.618)
    n_hist, bins_out, patches = ax.hist(peak_freqs, bins=bins, color='steelblue',
                                         edgecolor='white', alpha=0.85, linewidth=0.5)

    max_height = max(n_hist) if len(n_hist) > 0 else 1

    # Golden ratio scaled frequencies from f₀ = 7.6 Hz
    F0 = 7.6

    # Integer φ^n boundaries for EEG band shading
    phi_bounds = {
        -1: F0 * (PHI ** -1),  # ~4.7 Hz
        0: F0,                  # 7.6 Hz
        1: F0 * PHI,            # ~12.3 Hz
        2: F0 * (PHI ** 2),     # ~19.9 Hz
        3: F0 * (PHI ** 3),     # ~32.2 Hz
    }

    # EEG band shading using integer φ^n as boundaries (subtle muted colors)
    bands = [
        (1, phi_bounds[-1], 'Delta', '#d0d8e0'),
        (phi_bounds[-1], phi_bounds[0], 'Theta', '#e0dcd0'),
        (phi_bounds[0], phi_bounds[1], 'Alpha', '#d0e0d4'),
        (phi_bounds[1], phi_bounds[2], 'Beta-L', '#e0d4d0'),
        (phi_bounds[2], phi_bounds[3], 'Beta-H', '#dcd0e0'),
        (phi_bounds[3], 48, 'Gamma', '#d0dce0'),
    ]

    for low, high, band_name, color in bands:
        ax.axvspan(low, high, alpha=0.25, color=color, zorder=0)

    # Band labels flush at top using axes transform
    for low, high, band_name, color in bands:
        mid = (low + high) / 2
        # Convert x from data coords, y=1.0 in axes coords (very top)
        ax.text(mid, 0.99, band_name, ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='#444444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75,
                         edgecolor=color, linewidth=1.0))

    # φ^n reference lines with labels
    n_values = np.arange(-1, 4, 0.5)
    phi_freqs = [(n, F0 * (PHI ** n)) for n in n_values]

    for n, freq in phi_freqs:
        if 1 < freq < 48:
            is_integer = abs(n - round(n)) < 0.01
            style = '-' if is_integer else '--'
            alpha = 0.5 if is_integer else 0.3
            color = '#cc8800' if is_integer else '#bb5555'

            ax.axvline(freq, color=color, linestyle=style, alpha=alpha, linewidth=1.0, zorder=2)

            # Label just below band labels using axes transform
            if n == 0:
                label = f'f₀\n{freq:.1f}'
            elif is_integer:
                label = f'φ^{int(n)}\n{freq:.1f}'
            else:
                label = f'φ^{n:.1f}\n{freq:.1f}'
            ax.text(freq, 0.95, label, ha='center', va='top',
                   transform=ax.get_xaxis_transform(),
                   fontsize=7, color=color, fontweight='bold' if is_integer else 'normal', alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Peak Count', fontsize=12)

    short_name = os.path.basename(filename)
    if len(short_name) > 60:
        short_name = short_name[:57] + '...'

    ax.set_title(f'FOOOF Peak Frequency Distribution\n'
                 f'({len(peak_freqs)} peaks from {short_name}, {duration:.0f}s, 14 electrodes)',
                 fontsize=13, fontweight='bold')

    ax.set_xlim(0, 48)
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(np.arange(0, 50, 5))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def main():
    print("=" * 80)
    print("Per-File FOOOF Peak Frequency Histograms")
    print("=" * 80)
    print(f"Processing {len(ALL_FILES)} files...")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    results = []

    for i, (filepath, header) in enumerate(ALL_FILES):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(ALL_FILES)}] {filename}")

        if not os.path.exists(filepath):
            print(f"  File not found, skipping.")
            continue

        # Extract peaks
        peaks, duration = extract_peaks_from_file(filepath, header=header)

        if peaks is None or len(peaks) == 0:
            print(f"  No peaks detected, skipping.")
            continue

        peak_freqs = np.array([p['freq'] for p in peaks])

        # Create safe filename for output
        safe_name = filename.replace(' ', '_').replace('.csv', '')
        output_path = f"{OUTPUT_DIR}/{safe_name}_fooof_hist.png"

        # Generate histogram
        create_histogram(peak_freqs, filename, duration, output_path)

        print(f"  {len(peaks)} peaks detected, histogram saved.")

        results.append({
            'file': filename,
            'n_peaks': len(peaks),
            'duration': duration,
            'output': output_path
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'File':<55} {'Peaks':>8} {'Duration':>10}")
    print("-" * 75)

    for r in results:
        short_name = r['file'][:52] + '...' if len(r['file']) > 55 else r['file']
        print(f"{short_name:<55} {r['n_peaks']:>8} {r['duration']:>8.0f}s")

    print("-" * 75)
    print(f"Total: {len(results)} histograms saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

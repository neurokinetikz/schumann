"""Brain Invaders FOOOF Peak Analysis

Converts Brain Invaders EEG data to pipeline format, runs FOOOF peak detection,
and generates golden ratio analysis chart.
"""

import sys; sys.path.insert(0, './lib')
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof_harmonics import detect_harmonics_fooof

# Constants
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034
F0 = 7.6  # Base frequency for φ^n scaling
FS = 512  # Brain Invaders sampling rate (verified from timestamps)
EPOCH_SEC = 30  # Epoch duration in seconds for windowed FOOOF

# Brain Invaders electrode mapping
# Original: Time, Fp1, Fp2, F5, AFZ, F6, T7, Cz, T8, P7, P3, PZ, P4, P8, O1, Oz, O2, Event, Target
ORIGINAL_COLS = ['Time', 'Fp1', 'Fp2', 'F5', 'AFZ', 'F6', 'T7', 'Cz', 'T8',
                 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Event', 'Target']

# Electrodes to analyze (all 16 EEG channels)
ELECTRODES = ['EEG.Fp1', 'EEG.Fp2', 'EEG.F5', 'EEG.AFZ', 'EEG.F6', 'EEG.T7',
              'EEG.Cz', 'EEG.T8', 'EEG.P7', 'EEG.P3', 'EEG.PZ', 'EEG.P4',
              'EEG.P8', 'EEG.O1', 'EEG.Oz', 'EEG.O2']

DATA_DIR = 'data/brain_invaders'


def load_and_convert(filepath):
    """Load Brain Invaders CSV (headers already added)."""
    df = pd.read_csv(filepath)
    # Drop Event and Target columns if present
    df = df.drop(columns=['Event', 'Target'], errors='ignore')
    return df


def run_fooof_on_file(df, filename, all_peaks):
    """Run FOOOF on epochs within a single file."""
    n_samples = len(df)
    epoch_samples = EPOCH_SEC * FS  # 30 * 512 = 15,360 samples
    n_epochs = n_samples // epoch_samples

    print(f"  Processing {filename} ({n_epochs} epochs)...")
    peaks_before = len(all_peaks)

    for epoch_idx in range(n_epochs):
        start = epoch_idx * epoch_samples
        end = start + epoch_samples
        epoch_df = df.iloc[start:end].copy()

        for ch in ELECTRODES:
            if ch not in df.columns:
                continue
            try:
                harmonics, result = detect_harmonics_fooof(
                    records=epoch_df,
                    channels=[ch],
                    fs=FS,
                    freq_range=(1.0, 250.0),
                    max_n_peaks=50,
                    peak_threshold=0.001,
                    min_peak_height=0.0001,
                    peak_width_limits=(0.2, 20.0),
                )
                for p in result.all_peaks:
                    p['channel'] = ch
                    p['session'] = filename
                    p['epoch'] = epoch_idx
                    p['dataset'] = 'BRAIN_INVADERS'
                    all_peaks.append(p.copy())
            except Exception as e:
                pass  # Skip channels/epochs with errors

    return len(all_peaks) - peaks_before


def create_peak_distribution_chart(peak_freqs_arr, title, subtitle, output_path):
    """Create a peak frequency distribution chart with φ^n overlays."""
    fig, ax = plt.subplots(figsize=(20, 6))

    # Histogram with fine bins (0.309 Hz resolution)
    bins_fine = np.arange(1, 257, 0.309)
    n_hist, bins_out, patches = ax.hist(peak_freqs_arr, bins=bins_fine, color='steelblue',
                                         edgecolor='white', alpha=0.85, linewidth=0.5)

    max_height = max(n_hist) if len(n_hist) > 0 else 1

    # Integer φ^n boundaries for EEG band shading
    phi_bounds = {
        -1: F0 * (PHI ** -1),  # ~4.7 Hz
        0: F0,                  # 7.6 Hz
        1: F0 * PHI,            # ~12.3 Hz
        2: F0 * (PHI ** 2),     # ~19.9 Hz
        3: F0 * (PHI ** 3),     # ~32.2 Hz
        4: F0 * (PHI ** 4),     # ~52.1 Hz
        5: F0 * (PHI ** 5),     # ~84.3 Hz
        6: F0 * (PHI ** 6),     # ~136.4 Hz
        7: F0 * (PHI ** 7),     # ~220.7 Hz
    }

    # EEG band shading
    bands = [
        (1, phi_bounds[-1], 'Delta', '#d0d8e0'),
        (phi_bounds[-1], phi_bounds[0], 'Theta', '#e0dcd0'),
        (phi_bounds[0], phi_bounds[1], 'Alpha', '#d0e0d4'),
        (phi_bounds[1], phi_bounds[2], 'Beta-L', '#e0d4d0'),
        (phi_bounds[2], phi_bounds[3], 'Beta-H', '#dcd0e0'),
        (phi_bounds[3], phi_bounds[4], 'Gamma', '#d0dce0'),
        (phi_bounds[4], phi_bounds[5], 'Hi-γ', '#e0d0d8'),
        (phi_bounds[5], phi_bounds[6], 'Ultra-γ', '#d8e0d0'),
        (phi_bounds[6], phi_bounds[7], 'Hyper-γ', '#d0e0e0'),
        (phi_bounds[7], 256, 'ε-band', '#e0e0d0'),
    ]

    for low, high, band_name, color in bands:
        ax.axvspan(low, high, alpha=0.25, color=color, zorder=0)

    # Band labels
    for low, high, band_name, color in bands:
        mid = (low + high) / 2
        ax.text(mid, 0.99, band_name, ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='#444444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75,
                         edgecolor=color, linewidth=1.0))

    # φ^n reference lines with labels - including ALL position types
    # Position types: boundary (k), ¼ (k+0.25), 2° noble (k+0.382), attractor (k+0.5), 1° noble (k+0.618), ¾ (k+0.75)
    position_types = {
        'boundary':      {'offset': 0.0,   'color': '#cc8800', 'style': '-',  'alpha': 0.6, 'lw': 1.5, 'label': 'Boundary (φ^n)'},
        'quarter':       {'offset': 0.25,  'color': '#8888cc', 'style': '--', 'alpha': 0.3, 'lw': 0.8, 'label': '¼ (k+0.25)'},
        'noble_2':       {'offset': 0.382, 'color': '#88aa44', 'style': '--', 'alpha': 0.5, 'lw': 1.0, 'label': '2° Noble (k+0.382)'},
        'attractor':     {'offset': 0.5,   'color': '#cc4444', 'style': '--', 'alpha': 0.5, 'lw': 1.2, 'label': 'Attractor (k+0.5)'},
        'noble_1':       {'offset': 0.618, 'color': '#22aa88', 'style': ':',  'alpha': 0.6, 'lw': 1.5, 'label': '1° Noble (k+0.618)'},
        'three_quarter': {'offset': 0.75,  'color': '#888888', 'style': '--', 'alpha': 0.4, 'lw': 0.8, 'label': '¾ (k+0.75)'},
    }

    # Draw lines for each position type with legend entries
    legend_handles = []
    for ptype, props in position_types.items():
        first_line = True
        for k in range(-2, 8):  # Integer base positions (extended to φ^7)
            n = k + props['offset']
            freq = F0 * (PHI ** n)
            if 1 < freq < 256:
                line = ax.axvline(freq, color=props['color'], linestyle=props['style'],
                                  alpha=props['alpha'], linewidth=props['lw'], zorder=2)
                if first_line:
                    line.set_label(props['label'])
                    legend_handles.append(line)
                    first_line = False

                # Add frequency labels for key positions (boundary, attractor, nobles)
                if ptype in ['boundary', 'attractor', 'noble_1']:
                    if props['offset'] == 0:
                        if k == 0:
                            label = f'f₀\n{freq:.1f}'
                        else:
                            label = f'φ^{k}\n{freq:.1f}'
                    else:
                        label = f'{freq:.1f}'
                    ax.text(freq, 0.95, label, ha='center', va='top',
                           transform=ax.get_xaxis_transform(),
                           fontsize=6, color=props['color'], alpha=0.8)

    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Peak Count', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 256)
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(np.arange(0, 260, 20))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 90)
    print("Brain Invaders FOOOF Peak Analysis (256 Hz Extended)")
    print("=" * 90)
    print(f"Sampling Rate: {FS} Hz")
    print(f"FOOOF Range: 1-250 Hz (Nyquist: 256 Hz)")
    print(f"Electrodes: {len(ELECTRODES)}")

    # Find all subject CSV files
    csv_files = sorted([f for f in os.listdir(DATA_DIR)
                       if f.startswith('subject_') and f.endswith('.csv')])
    print(f"Found {len(csv_files)} subject files\n")

    all_peaks = []
    processed = 0
    start_time = time.time()

    for i, filename in enumerate(csv_files):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            # Load and convert
            df = load_and_convert(filepath)

            # Run FOOOF
            n_peaks = run_fooof_on_file(df, filename, all_peaks)
            processed += 1

            # Progress reporting
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(csv_files) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  {i+1}/{len(csv_files)} | {filename} | {n_peaks} peaks | "
                  f"{len(all_peaks)} total | {rate:.2f} files/sec | ~{remaining:.1f} min left")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"\n{'=' * 90}")
    print(f"SUMMARY")
    print(f"{'=' * 90}")
    print(f"Processed: {processed} files")
    print(f"Total peaks: {len(all_peaks)}")

    if len(all_peaks) == 0:
        print("No peaks detected!")
        return

    # Save peaks CSV
    peaks_df = pd.DataFrame(all_peaks)
    peaks_csv = 'golden_ratio_peaks_BRAIN_INVADERS_256Hz.csv'
    peaks_df.to_csv(peaks_csv, index=False)
    print(f"Saved peaks: {peaks_csv}")

    # Generate chart
    print(f"\n{'=' * 90}")
    print("GENERATING VISUALIZATION...")
    print(f"{'=' * 90}")

    peak_freqs = np.array([p['freq'] for p in all_peaks])
    # Filter to visible range (avoid edge effects)
    peak_freqs_filtered = peak_freqs[(peak_freqs >= 1) & (peak_freqs <= 250)]
    n_sessions = len(set(p['session'] for p in all_peaks))
    n_epochs = len(set((p['session'], p['epoch']) for p in all_peaks))

    create_peak_distribution_chart(
        peak_freqs_filtered,
        'FOOOF Peak Frequency Distribution - BRAIN_INVADERS (256 Hz)',
        f'({len(peak_freqs_filtered)} peaks across {n_sessions} subjects, {n_epochs} epochs, {len(ELECTRODES)} electrodes)',
        'golden_ratio_peaks_BRAIN_INVADERS_256Hz.png'
    )

    print("\nDone!")


if __name__ == '__main__':
    main()

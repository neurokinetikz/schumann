"""Golden Ratio Peak Frequency Analysis - EMOTIONS Dataset

Detects FOOOF peaks from EEG data and analyzes frequency distributions
for the EMOTIONS dataset (256 Hz sampling rate).

Samples files evenly for faster processing while maintaining representativeness.
"""

import sys; sys.path.insert(0, './lib')
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import utilities
from fooof_harmonics import detect_harmonics_fooof
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034
FS = 256  # EMOTIONS dataset is 256 Hz
F0 = 7.6  # Base frequency for φ^n scaling

# Electrode configuration (EPOC X)
EPOC_ELECTRODES = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1',
                   'EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']

# Sampling configuration
MAX_FILES = None  # Process all files


def list_csv_files(directory):
    """List all CSV files in a directory."""
    return sorted(glob.glob(os.path.join(directory, "*.csv")))


def analyze_file(filepath, header=0, electrodes=None, fs=256):
    """Analyze a single EEG file for peak frequencies."""
    if electrodes is None:
        electrodes = EPOC_ELECTRODES

    filename = os.path.basename(filepath)

    # Load EEG data
    try:
        RECORDS = utilities.load_eeg_csv(filepath, electrodes=electrodes, fs=fs, header=header, device='emotiv')
    except Exception as e:
        return None, []

    duration = len(RECORDS) / fs

    # Run FOOOF per channel
    all_peaks = []
    for ch in electrodes:
        try:
            harmonics, result = detect_harmonics_fooof(
                records=RECORDS,
                channels=[ch],
                fs=fs,
                freq_range=(1.0, 120.0),
                max_n_peaks=40,
                peak_threshold=0.001,
                min_peak_height=0.0001,
                peak_width_limits=(0.2, 20.0),
            )
            for p in result.all_peaks:
                p['channel'] = ch
                p['session'] = filename
                all_peaks.append(p)
        except:
            pass

    if len(all_peaks) < 2:
        return None, []

    result = {
        'file': filename,
        'duration': duration,
        'n_peaks': len(all_peaks),
        'electrodes': len(electrodes),
    }

    return result, all_peaks


def create_peak_distribution_chart(peak_freqs_arr, title, subtitle, output_path):
    """Create a peak frequency distribution chart with φ^n overlays."""
    fig, ax = plt.subplots(figsize=(18, 6))

    # Histogram with fine bins (0.309 Hz resolution)
    bins_fine = np.arange(1, 121, 0.309)
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
    }

    # EEG band shading using integer φ^n as boundaries (subtle muted colors)
    bands = [
        (1, phi_bounds[-1], 'Delta', '#d0d8e0'),
        (phi_bounds[-1], phi_bounds[0], 'Theta', '#e0dcd0'),
        (phi_bounds[0], phi_bounds[1], 'Alpha', '#d0e0d4'),
        (phi_bounds[1], phi_bounds[2], 'Beta-L', '#e0d4d0'),
        (phi_bounds[2], phi_bounds[3], 'Beta-H', '#dcd0e0'),
        (phi_bounds[3], phi_bounds[4], 'Gamma', '#d0dce0'),
        (phi_bounds[4], phi_bounds[5], 'Hi-γ', '#e0d0d8'),
        (phi_bounds[5], 120, 'Ultra-γ', '#d8e0d0'),
    ]

    for low, high, band_name, color in bands:
        ax.axvspan(low, high, alpha=0.25, color=color, zorder=0)

    # Band labels flush at top using axes transform
    for low, high, band_name, color in bands:
        mid = (low + high) / 2
        ax.text(mid, 0.99, band_name, ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='#444444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75,
                         edgecolor=color, linewidth=1.0))

    # φ^n reference lines with labels - including ALL position types
    # Nobles are symmetric around attractor (0.5): k+x and k+(1-x)
    position_types = {
        'noble_4':       {'offset': 0.146, 'color': '#aa88cc', 'style': ':',  'alpha': 0.25, 'lw': 0.6, 'label': '4° Noble (k+0.146)'},
        'noble_3':       {'offset': 0.236, 'color': '#88ccaa', 'style': ':',  'alpha': 0.3,  'lw': 0.7, 'label': '3° Noble (k+0.236)'},
        'quarter':       {'offset': 0.25,  'color': '#8888cc', 'style': '--', 'alpha': 0.3,  'lw': 0.8, 'label': '¼ (k+0.25)'},
        'noble_2':       {'offset': 0.382, 'color': '#88aa44', 'style': '--', 'alpha': 0.5,  'lw': 1.0, 'label': '2° Noble (k+0.382)'},
        'boundary':      {'offset': 0.0,   'color': '#cc8800', 'style': '-',  'alpha': 0.6,  'lw': 1.5, 'label': 'Boundary (φ^n)'},
        'attractor':     {'offset': 0.5,   'color': '#cc4444', 'style': '--', 'alpha': 0.5,  'lw': 1.2, 'label': 'Attractor (k+0.5)'},
        'noble_1':       {'offset': 0.618, 'color': '#22aa88', 'style': ':',  'alpha': 0.6,  'lw': 1.5, 'label': '1° Noble (k+0.618)'},
        'three_quarter': {'offset': 0.75,  'color': '#888888', 'style': '--', 'alpha': 0.4,  'lw': 0.8, 'label': '¾ (k+0.75)'},
        'noble_3_inv':   {'offset': 0.764, 'color': '#88ccaa', 'style': ':',  'alpha': 0.3,  'lw': 0.7, 'label': '3° Inv (k+0.764)'},
        'noble_4_inv':   {'offset': 0.854, 'color': '#aa88cc', 'style': ':',  'alpha': 0.25, 'lw': 0.6, 'label': '4° Inv (k+0.854)'},
    }

    # Draw lines for each position type with legend entries
    legend_handles = []
    for ptype, props in position_types.items():
        first_line = True
        for k in range(-1, 6):  # Integer base positions (extended to φ^5)
            n = k + props['offset']
            freq = F0 * (PHI ** n)
            if 1 < freq < 120:
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
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.9)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Peak Count', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(np.arange(0, 125, 10))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


def main():
    # Get EMOTIONS CSV files
    emotions_files = list_csv_files("data/emotions")
    print(f"Found {len(emotions_files)} EMOTIONS CSV files")

    if not emotions_files:
        print("No CSV files found in data/emotions/")
        return

    # Sample files if needed
    if MAX_FILES and len(emotions_files) > MAX_FILES:
        # Sample evenly across the file list
        indices = np.linspace(0, len(emotions_files) - 1, MAX_FILES, dtype=int)
        emotions_files = [emotions_files[i] for i in indices]
        print(f"Sampled {len(emotions_files)} files for analysis")

    # Process files with multiprocessing
    print(f"\n{'='*90}")
    print(f"PROCESSING EMOTIONS DATASET (256 Hz) - {len(emotions_files)} files")
    print(f"{'='*90}")

    # Process files sequentially (single-threaded for reliability)
    print("Processing files sequentially (single-threaded)...")

    all_session_peaks = []
    all_results = []
    start_time = time.time()

    for i, filepath in enumerate(emotions_files):
        result, peaks = analyze_file(filepath, header=0, electrodes=EPOC_ELECTRODES, fs=FS)

        if result:
            all_results.append(result)
            all_session_peaks.extend(peaks)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(emotions_files) - i - 1) / rate / 60
            print(f"  {i+1}/{len(emotions_files)} files | {len(all_session_peaks)} peaks | {rate:.1f} files/sec | ~{remaining:.1f} min left")

    print(f"\n{'='*90}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*90}")
    print(f"Processed: {len(all_results)}/{len(emotions_files)} files successfully")
    print(f"Total peaks detected: {len(all_session_peaks)}")

    if len(all_session_peaks) == 0:
        print("No peaks detected!")
        return

    # Export peaks to CSV
    peaks_df = pd.DataFrame(all_session_peaks)
    peaks_csv_path = 'golden_ratio_peaks_EMOTIONS.csv'
    peaks_df.to_csv(peaks_csv_path, index=False)
    print(f"\nSaved peaks: {peaks_csv_path} ({len(peaks_df)} rows)")

    # Generate visualization
    print(f"\n{'='*90}")
    print("GENERATING VISUALIZATION...")
    print(f"{'='*90}")

    peak_freqs = np.array([p['freq'] for p in all_session_peaks])
    n_sessions = len(set(p['session'] for p in all_session_peaks))

    output_path = 'golden_ratio_peaks_EMOTIONS.png'
    create_peak_distribution_chart(
        peak_freqs,
        'FOOOF Peak Frequency Distribution - EMOTIONS',
        f'({len(all_session_peaks)} peaks across {n_sessions} sessions, {len(EPOC_ELECTRODES)} electrodes)',
        output_path
    )
    print(f"Saved: {output_path}")

    # Print summary statistics
    print(f"\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}")
    print(f"Total files processed: {len(all_results)}")
    print(f"Total peaks: {len(all_session_peaks)}")
    print(f"Mean peaks per session: {len(all_session_peaks)/max(len(all_results),1):.1f}")
    print(f"Frequency range: {peak_freqs.min():.2f} - {peak_freqs.max():.2f} Hz")
    print(f"Median frequency: {np.median(peak_freqs):.2f} Hz")


if __name__ == '__main__':
    main()

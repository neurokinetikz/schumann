"""Golden Ratio Peak Frequency Analysis - Multi-File

Detects FOOOF peaks from EEG data and analyzes pairwise frequency ratios
for proximity to golden ratio (φ) scaling across multiple sessions.
"""

import sys; sys.path.insert(0, './lib')
import numpy as np
import pandas as pd
from itertools import combinations
import os
import utilities
from fooof_harmonics import detect_harmonics_fooof

# Constants
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618034
FS = 128

# Electrode configurations by device
EPOC_ELECTRODES = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1',
                   'EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']
INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

# Default electrodes and device (can be overridden per dataset)
ELECTRODES = EPOC_ELECTRODES
DEVICE = 'emotiv'  # 'emotiv' or 'muse'

def list_csv_files(directory):
    """List all CSV files in a directory."""
    return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

# Original FILES with header=1
FILES = [
    ('data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv', 1),
    ('data/Test_06.11.20_14.28.18.md.pm.bp.csv', 1),
    ('data/20201229_29.12.20_11.27.57.md.pm.bp.csv', 1),
    ('data/med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv', 1),
    ('data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv', 1),
]

# PHYSF files with header=0
PHYSF = [(f, 0) for f in list_csv_files("data/PhySF")]

# VEP files with header=0
VEP = [(f, 0) for f in list_csv_files("data/vep")]

# MPENG files with header=0 (non-EEG files moved to other/ subdirectory)
MPENG1 = [(f, 0) for f in list_csv_files("data/mpeng1")]
MPENG2 = [(f, 0) for f in list_csv_files("data/mpeng2")]

# ArEEG files with header=1 (metadata on line 1, columns on line 2)
ArEEG = [(f, 1) for f in list_csv_files("data/ArEEG")]

# Insight files with header=1 (metadata on line 1, columns on line 2)
INSIGHT = [(f, 1) for f in list_csv_files("data/insight")]

# Muse files with header=0 (columns eeg_1-4 mapped to EEG.AF7, AF8, TP9, TP10)
MUSE = [(f, 0) for f in list_csv_files("data/muse")]

# Dataset configurations: list of (name, files_list, electrodes, device) tuples
# Each dataset is processed with its own electrode configuration, then peaks are combined
DATASET_CONFIGS = [
    ('FILES', FILES, EPOC_ELECTRODES, 'emotiv'),
    ('INSIGHT', INSIGHT, INSIGHT_ELECTRODES, 'emotiv'),
    ('PHYSF', PHYSF, EPOC_ELECTRODES, 'emotiv'),
    ('VEP', VEP, EPOC_ELECTRODES, 'emotiv'),
    # ('ArEEG', ArEEG, EPOC_ELECTRODES, 'emotiv'),
    ('MPENG1', MPENG1, EPOC_ELECTRODES, 'emotiv'),
    ('MPENG2', MPENG2, EPOC_ELECTRODES, 'emotiv'),
    ('MUSE', MUSE, MUSE_ELECTRODES, 'muse'),
]
DATASET_NAME = "ALL"

# For single-dataset runs, use one of these instead:
# DATASET_CONFIGS = [('FILES', FILES, EPOC_ELECTRODES, 'emotiv')]; DATASET_NAME = "FILES"
# DATASET_CONFIGS = [('PHYSF', PHYSF, EPOC_ELECTRODES, 'emotiv')]; DATASET_NAME = "PHYSF"
# DATASET_CONFIGS = [('VEP', VEP, EPOC_ELECTRODES, 'emotiv')]; DATASET_NAME = "VEP"
# DATASET_CONFIGS = [('MPENG', MPENG1 + MPENG2, EPOC_ELECTRODES, 'emotiv')]; DATASET_NAME = "MPENG"
# DATASET_CONFIGS = [('INSIGHT', INSIGHT, INSIGHT_ELECTRODES, 'emotiv')]; DATASET_NAME = "INSIGHT"
# DATASET_CONFIGS = [('MUSE', MUSE, MUSE_ELECTRODES, 'muse')]; DATASET_NAME = "MUSE"
# DATASET_CONFIGS = [('ArEEG', ArEEG, EPOC_ELECTRODES, 'emotiv')]; DATASET_NAME = "ArEEG"

def find_nearest_phi_power(ratio, max_n=6):
    """Find nearest φ^n where n is integer or half-integer."""
    n_values = np.arange(-max_n, max_n + 0.5, 0.5)
    phi_powers = PHI ** n_values
    distances = np.abs(ratio - phi_powers)
    best_idx = np.argmin(distances)
    return {
        'nearest_n': n_values[best_idx],
        'phi_power': phi_powers[best_idx],
        'distance': distances[best_idx],
        'rel_error': distances[best_idx] / phi_powers[best_idx]
    }

def analyze_file(filepath, all_session_peaks, header=1, electrodes=None, device='emotiv'):
    """Analyze a single EEG file for golden ratio relationships."""
    if electrodes is None:
        electrodes = EPOC_ELECTRODES

    filename = os.path.basename(filepath)

    # Load EEG data
    try:
        RECORDS = utilities.load_eeg_csv(filepath, electrodes=electrodes, fs=FS, header=header, device=device)
    except Exception as e:
        print(f"  Error loading: {e}")
        return None

    duration = len(RECORDS) / FS

    # Run FOOOF per channel
    all_peaks = []
    for ch in electrodes:
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
        except:
            pass

    if len(all_peaks) < 2:
        return None

    # Add to global collection with session info
    for p in all_peaks:
        p['session'] = filename
        all_session_peaks.append(p.copy())

    # Group peaks by channel
    peaks_by_channel = {}
    for p in all_peaks:
        ch = p['channel']
        if ch not in peaks_by_channel:
            peaks_by_channel[ch] = []
        peaks_by_channel[ch].append(p)

    # Analyze each channel
    channel_results = []
    for ch in electrodes:
        if ch not in peaks_by_channel or len(peaks_by_channel[ch]) < 2:
            continue

        ch_peaks = peaks_by_channel[ch]
        ch_freqs = sorted([p['freq'] for p in ch_peaks])
        n_pairs = len(ch_freqs) * (len(ch_freqs) - 1) // 2

        # Compute pairwise ratios
        within_1pct = 0
        within_2pct = 0
        within_5pct = 0
        best_error = float('inf')
        best_match = None

        for f1, f2 in combinations(ch_freqs, 2):
            ratio = f1 / f2 if f1 > f2 else f2 / f1
            phi_match = find_nearest_phi_power(ratio)
            rel_error = phi_match['rel_error'] * 100

            if rel_error < 1.0:
                within_1pct += 1
            if rel_error < 2.0:
                within_2pct += 1
            if rel_error < 5.0:
                within_5pct += 1

            if rel_error < best_error:
                best_error = rel_error
                best_match = {
                    'freq_high': max(f1, f2),
                    'freq_low': min(f1, f2),
                    'ratio': ratio,
                    'phi_n': phi_match['nearest_n'],
                    'error': rel_error
                }

        channel_results.append({
            'channel': ch.replace('EEG.', ''),
            'n_peaks': len(ch_peaks),
            'n_pairs': n_pairs,
            'within_1pct': within_1pct,
            'within_2pct': within_2pct,
            'within_5pct': within_5pct,
            'best_match': best_match
        })

    # Aggregate
    total_pairs = sum(r['n_pairs'] for r in channel_results)
    total_1pct = sum(r['within_1pct'] for r in channel_results)
    total_5pct = sum(r['within_5pct'] for r in channel_results)

    return {
        'file': filename,
        'duration': duration,
        'total_peaks': len(all_peaks),
        'total_pairs': total_pairs,
        'within_1pct': total_1pct,
        'within_5pct': total_5pct,
        'pct_1': 100 * total_1pct / total_pairs if total_pairs > 0 else 0,
        'pct_5': 100 * total_5pct / total_pairs if total_pairs > 0 else 0,
        'channel_results': channel_results
    }

# Main execution
print("=" * 90)
print("Golden Ratio (φ) Peak Frequency Analysis - Multi-Session")
print("=" * 90)
print(f"φ = {PHI:.6f}")
total_files = sum(len(files) for _, files, _, _ in DATASET_CONFIGS)
print(f"Analyzing {total_files} files across {len(DATASET_CONFIGS)} dataset(s) - {DATASET_NAME}...\n")

all_results = []
all_session_peaks = []  # Collect ALL peaks across ALL sessions
peaks_by_dataset = {}  # Collect peaks per dataset
total_electrodes = set()  # Track unique electrodes used

for dataset_name, files_list, electrodes, device in DATASET_CONFIGS:
    total_electrodes.update(electrodes)
    peaks_by_dataset[dataset_name] = []  # Initialize list for this dataset
    for filepath, header in files_list:
        filename = os.path.basename(filepath)
        print(f"\n{'─'*90}")
        print(f"FILE: {filename}")
        print(f"{'─'*90}")

        if not os.path.exists(filepath):
            print(f"  File not found!")
            continue

        prev_count = len(all_session_peaks)
        result = analyze_file(filepath, all_session_peaks, header=header, electrodes=electrodes, device=device)

        # Tag new peaks with dataset name and add to per-dataset collection
        for p in all_session_peaks[prev_count:]:
            p['dataset'] = dataset_name
            peaks_by_dataset[dataset_name].append(p)

        if result is None:
            print(f"  Could not analyze file")
            continue

        all_results.append(result)

        # Print per-channel summary for this file
        print(f"  Duration: {result['duration']:.1f}s | Peaks: {result['total_peaks']} | Pairs: {result['total_pairs']}")
        print(f"  Within 1% of φ^n: {result['within_1pct']} ({result['pct_1']:.1f}%)")
        print(f"  Within 5% of φ^n: {result['within_5pct']} ({result['pct_5']:.1f}%)")

        # Best matches per channel
        print(f"\n  Best φ^n matches by channel:")
        for cr in result['channel_results']:
            if cr['best_match']:
                bm = cr['best_match']
                print(f"    {cr['channel']:<4}: {bm['freq_high']:6.2f}/{bm['freq_low']:6.2f} = φ^{bm['phi_n']:.1f} ({bm['error']:.3f}%)")

# Grand summary
print(f"\n{'='*90}")
print("GRAND SUMMARY ACROSS ALL FILES")
print(f"{'='*90}")
print(f"{'File':<45} {'Dur':>6} {'Peaks':>6} {'Pairs':>6} {'<1%':>6} {'<5%':>6} {'%<1%':>7} {'%<5%':>7}")
print(f"{'-'*90}")

grand_pairs = 0
grand_1pct = 0
grand_5pct = 0

for r in all_results:
    short_name = r['file'][:42] + '...' if len(r['file']) > 45 else r['file']
    print(f"{short_name:<45} {r['duration']:>5.0f}s {r['total_peaks']:>6} {r['total_pairs']:>6} "
          f"{r['within_1pct']:>6} {r['within_5pct']:>6} {r['pct_1']:>6.1f}% {r['pct_5']:>6.1f}%")
    grand_pairs += r['total_pairs']
    grand_1pct += r['within_1pct']
    grand_5pct += r['within_5pct']

print(f"{'-'*90}")
print(f"{'TOTAL':<45} {'':>6} {'':>6} {grand_pairs:>6} {grand_1pct:>6} {grand_5pct:>6} "
      f"{100*grand_1pct/grand_pairs:>6.1f}% {100*grand_5pct/grand_pairs:>6.1f}%")

print(f"\n{'='*90}")
print("CONCLUSION")
print(f"{'='*90}")
print(f"Across {len(all_results)} sessions and {grand_pairs} peak pairs:")
print(f"  • {grand_1pct} pairs ({100*grand_1pct/grand_pairs:.1f}%) within 1% of φ^n")
print(f"  • {grand_5pct} pairs ({100*grand_5pct/grand_pairs:.1f}%) within 5% of φ^n")

# =============================================================================
# PEAK FREQUENCY CLUSTERING ANALYSIS
# =============================================================================
print(f"\n{'='*90}")
print("PEAK FREQUENCY CLUSTERING (All Sessions, All Electrodes)")
print(f"{'='*90}")
print(f"Total peaks collected: {len(all_session_peaks)}")

# Create histogram with 1 Hz bins
peak_freqs = np.array([p['freq'] for p in all_session_peaks])
bins = np.arange(1, 51, 1)  # 1 Hz bins from 1-50 Hz
hist, edges = np.histogram(peak_freqs, bins=bins)

# Find top clusters
bin_counts = [(edges[i], hist[i]) for i in range(len(hist)) if hist[i] > 0]
bin_counts_sorted = sorted(bin_counts, key=lambda x: -x[1])

print(f"\nFrequency Distribution (1 Hz bins):")
print(f"{'Freq (Hz)':<12} {'Count':>6} {'Bar':<50}")
print(f"{'-'*70}")

max_count = max(hist)
for i in range(len(hist)):
    if hist[i] > 0:
        freq = edges[i]
        count = hist[i]
        bar_len = int(50 * count / max_count)
        bar = '█' * bar_len
        # Mark Schumann harmonics
        sr_marker = ''
        for sr in [7.83, 14.3, 20.8, 27.3, 33.8, 40.3]:
            if freq <= sr < freq + 1:
                sr_marker = f' ← SR {sr:.1f}'
                break
        print(f"{freq:>5.0f}-{freq+1:<5.0f} {count:>6} {bar}{sr_marker}")

# Top 10 frequency clusters
print(f"\n{'='*90}")
print("TOP 15 PEAK FREQUENCY CLUSTERS")
print(f"{'='*90}")
print(f"{'Rank':<6} {'Freq Range':<12} {'Count':>6} {'% of Total':>12}")
print(f"{'-'*40}")
for i, (freq, count) in enumerate(bin_counts_sorted[:15]):
    pct = 100 * count / len(all_session_peaks)
    print(f"{i+1:<6} {freq:>4.0f}-{freq+1:<4.0f} Hz {count:>6} {pct:>11.1f}%")

# Check if clusters follow φ relationships
print(f"\n{'='*90}")
print("CLUSTER φ^n RELATIONSHIPS")
print(f"{'='*90}")
top_freqs = [f + 0.5 for f, c in bin_counts_sorted[:10]]  # Use bin centers
print(f"Top cluster centers: {[f'{f:.1f}' for f in top_freqs]}")
print(f"\nRatios between top clusters:")
for i, f1 in enumerate(top_freqs[:6]):
    for f2 in top_freqs[i+1:8]:
        if f1 > f2:
            ratio = f1 / f2
        else:
            ratio = f2 / f1
        phi_match = find_nearest_phi_power(ratio)
        if phi_match['rel_error'] < 0.05:  # Within 5%
            print(f"  {max(f1,f2):.1f}/{min(f1,f2):.1f} = {ratio:.3f} ≈ φ^{phi_match['nearest_n']:.1f} ({phi_match['rel_error']*100:.1f}%)")

# =============================================================================
# VISUALIZATION: Peak Frequency Distribution
# =============================================================================
import matplotlib.pyplot as plt

# =============================================================================
# CSV EXPORT
# =============================================================================
print(f"\n{'='*90}")
print("EXPORTING CSV FILES...")
print(f"{'='*90}")

# Export per-file summary
summary_df = pd.DataFrame([{
    'file': r['file'],
    'duration_sec': r['duration'],
    'total_peaks': r['total_peaks'],
    'total_pairs': r['total_pairs'],
    'within_1pct': r['within_1pct'],
    'within_5pct': r['within_5pct'],
    'pct_within_1pct': r['pct_1'],
    'pct_within_5pct': r['pct_5'],
} for r in all_results])
summary_csv_path = f'golden_ratio_summary_{DATASET_NAME}.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"Saved per-file summary: {summary_csv_path} ({len(summary_df)} rows)")

# Export all detected peaks across sessions
peaks_df = pd.DataFrame(all_session_peaks)
peaks_csv_path = f'golden_ratio_peaks_{DATASET_NAME}.csv'
peaks_df.to_csv(peaks_csv_path, index=False)
print(f"Saved all peaks: {peaks_csv_path} ({len(peaks_df)} rows)")

# Export aggregate peaks per dataset
print(f"\nPer-dataset peak exports:")
for ds_name, ds_peaks in peaks_by_dataset.items():
    if len(ds_peaks) > 0:
        ds_df = pd.DataFrame(ds_peaks)
        ds_csv_path = f'golden_ratio_peaks_{ds_name}.csv'
        ds_df.to_csv(ds_csv_path, index=False)
        print(f"  Saved {ds_name}: {ds_csv_path} ({len(ds_df)} rows)")

# =============================================================================
# VISUALIZATION
# =============================================================================
print(f"\n{'='*90}")
print("GENERATING VISUALIZATIONS...")
print(f"{'='*90}")

# Golden ratio scaling from f₀ = 7.6 Hz
F0 = 7.6

def create_peak_distribution_chart(peak_freqs_arr, title, subtitle, output_path):
    """Create a peak frequency distribution chart with φ^n overlays."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram with fine bins (0.309 Hz resolution)
    bins_fine = np.arange(1, 51, 0.309)
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
        for k in range(-2, 4):  # Integer base positions (start at -2 to include delta band)
            n = k + props['offset']
            freq = F0 * (PHI ** n)
            if 1 < freq < 48:
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
    ax.set_xlim(0, 48)
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(np.arange(0, 50, 5))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

# Generate per-dataset charts
print("\nPer-dataset visualizations:")
for ds_name, ds_peaks in peaks_by_dataset.items():
    if len(ds_peaks) > 0:
        ds_freqs = np.array([p['freq'] for p in ds_peaks])
        n_sessions = len(set(p['session'] for p in ds_peaks))
        output_path = f'golden_ratio_peaks_{ds_name}.png'
        create_peak_distribution_chart(
            ds_freqs,
            f'FOOOF Peak Frequency Distribution - {ds_name}',
            f'({len(ds_peaks)} peaks across {n_sessions} sessions)',
            output_path
        )
        print(f"  Saved: {output_path}")

# Generate total aggregate chart
print("\nTotal aggregate visualization:")
output_path = f'golden_ratio_peaks_{DATASET_NAME}.png'
create_peak_distribution_chart(
    peak_freqs,
    f'FOOOF Peak Frequency Distribution - {DATASET_NAME}',
    f'({len(all_session_peaks)} peaks across {len(all_results)} sessions, {len(total_electrodes)} electrodes)',
    output_path
)
print(f"  Saved: {output_path}")

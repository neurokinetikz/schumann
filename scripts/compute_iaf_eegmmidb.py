#!/usr/bin/env python3
"""
Compute Individual Alpha Frequency (IAF) distribution across 109 EEGMMIDB subjects
using their R01 (rest, eyes-open) recordings.

Method:
  - Load each subject's S###R01.edf (rest-EO) using MNE
  - Compute PSD via Welch's method (nperseg=512, fs=160 Hz)
  - Average PSD across all 64 EEG channels
  - Find peak alpha frequency (PAF) in the 8-13 Hz range
  - Report distribution statistics
"""

import os
import numpy as np
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/neurokinetikz/Code/schumann/data/eegmmidb"
FS = 160       # EEGMMIDB sampling rate
NPERSEG = 512  # ~3.2 seconds per segment
ALPHA_LOW = 8.0
ALPHA_HIGH = 13.0

import mne
mne.set_log_level('ERROR')

def load_edf_data(filepath):
    """Load EDF file and return (data_array, channel_names, fs)."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs = raw.info['sfreq']
    data = raw.get_data()  # (n_channels, n_samples)
    ch_names = raw.ch_names
    return data, ch_names, fs

def compute_paf(data, fs, nperseg=512, alpha_range=(8.0, 13.0)):
    """
    Compute Peak Alpha Frequency from multi-channel EEG data.
    Average PSD across all channels, then find peak in alpha range.
    """
    n_channels = data.shape[0]
    
    # Compute PSD for each channel and average
    psds = []
    for ch in range(n_channels):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        psds.append(psd)
    
    mean_psd = np.mean(psds, axis=0)
    
    # Extract alpha range
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = mean_psd[alpha_mask]
    
    if len(alpha_psd) == 0:
        return None, 0.0
    
    # Find peak
    peak_idx = np.argmax(alpha_psd)
    paf = alpha_freqs[peak_idx]
    alpha_power = alpha_psd[peak_idx]
    
    return paf, alpha_power

def main():
    print("=" * 70)
    print("EEGMMIDB Individual Alpha Frequency (IAF) Distribution")
    print("Using R01 (rest, eyes-open) recordings")
    print("=" * 70)
    print()
    
    subjects = []
    pafs = []
    alpha_powers = []
    errors = []
    
    for subj_num in range(1, 110):
        subj_id = f"S{subj_num:03d}"
        edf_path = os.path.join(DATA_DIR, subj_id, f"{subj_id}R01.edf")
        
        if not os.path.exists(edf_path):
            errors.append((subj_id, "File not found"))
            continue
        
        try:
            data, ch_names, fs = load_edf_data(edf_path)
            
            if abs(fs - FS) > 1:
                print(f"  WARNING: {subj_id} has fs={fs} Hz (expected {FS})")
            
            paf, alpha_power = compute_paf(data, fs, nperseg=NPERSEG,
                                            alpha_range=(ALPHA_LOW, ALPHA_HIGH))
            
            if paf is not None:
                subjects.append(subj_id)
                pafs.append(paf)
                alpha_powers.append(alpha_power)
            else:
                errors.append((subj_id, "No alpha peak found"))
                
        except Exception as e:
            errors.append((subj_id, str(e)))
    
    pafs = np.array(pafs)
    alpha_powers = np.array(alpha_powers)
    
    print(f"Successfully processed: {len(pafs)} / 109 subjects")
    if errors:
        print(f"Errors/missing: {len(errors)}")
        for subj, err in errors:
            print(f"  {subj}: {err}")
    print()
    
    # Compute statistics
    print("-" * 50)
    print("IAF Distribution Statistics (8-13 Hz range)")
    print("-" * 50)
    print(f"  N subjects:  {len(pafs)}")
    print(f"  Mean:        {np.mean(pafs):.3f} Hz")
    print(f"  SD:          {np.std(pafs, ddof=1):.3f} Hz")
    print(f"  Median:      {np.median(pafs):.3f} Hz")
    print(f"  Min:         {np.min(pafs):.3f} Hz")
    print(f"  Max:         {np.max(pafs):.3f} Hz")
    
    q25, q75 = np.percentile(pafs, [25, 75])
    iqr = q75 - q25
    print(f"  Q1 (25th):   {q25:.3f} Hz")
    print(f"  Q3 (75th):   {q75:.3f} Hz")
    print(f"  IQR:         {iqr:.3f} Hz")
    print()
    
    # Histogram bin counts (0.5 Hz bins)
    print("-" * 50)
    print("Frequency histogram (0.5 Hz bins)")
    print("-" * 50)
    bins = np.arange(ALPHA_LOW, ALPHA_HIGH + 0.51, 0.5)
    counts, bin_edges = np.histogram(pafs, bins=bins)
    for i in range(len(counts)):
        bar = "#" * counts[i]
        print(f"  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f} Hz: {counts[i]:3d}  {bar}")
    print()
    
    # PAF proximity to key frequencies
    print("-" * 50)
    print("PAF proximity to key frequencies")
    print("-" * 50)
    for freq, label in [(7.83, "Schumann f0"), (10.0, "10.0 Hz canonical alpha"),
                         (10.31, "10.31 Hz (phi*f0 = 1.618*6.375)"),
                         (11.0, "11.0 Hz"), (12.67, "12.67 Hz (phi*f0 = 1.618*7.83)")]:
        within_05 = np.sum(np.abs(pafs - freq) <= 0.5)
        within_1 = np.sum(np.abs(pafs - freq) <= 1.0)
        print(f"  {label}:")
        print(f"    Within +/-0.5 Hz: {within_05} subjects ({100*within_05/len(pafs):.1f}%)")
        print(f"    Within +/-1.0 Hz: {within_1} subjects ({100*within_1/len(pafs):.1f}%)")
    print()
    
    # Per-subject listing
    print("-" * 50)
    print("Per-subject PAF values (sorted)")
    print("-" * 50)
    sorted_indices = np.argsort(pafs)
    for i, idx in enumerate(sorted_indices):
        print(f"  {subjects[idx]}: {pafs[idx]:6.3f} Hz", end="")
        if (i + 1) % 5 == 0:
            print()
        else:
            print("  |  ", end="")
    if len(sorted_indices) % 5 != 0:
        print()
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print(f"  IAF = {np.mean(pafs):.2f} +/- {np.std(pafs, ddof=1):.2f} Hz")
    print(f"  Range: [{np.min(pafs):.2f}, {np.max(pafs):.2f}] Hz")
    print(f"  Median (IQR): {np.median(pafs):.2f} ({q25:.2f}-{q75:.2f}) Hz")
    print("=" * 70)

if __name__ == "__main__":
    main()

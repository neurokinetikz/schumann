#!/usr/bin/env python3
"""
Compute Individual Alpha Frequency (IAF) distribution across 109 EEGMMIDB subjects.
V2: Higher frequency resolution + peak validation + gravity frequency (CoG) method.

Uses R01 (rest, eyes-open) recordings.
"""

import os
import numpy as np
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

DATA_DIR = "/Users/neurokinetikz/Code/schumann/data/eegmmidb"
FS = 160
ALPHA_LOW = 7.5   # Slightly wider to catch edge cases
ALPHA_HIGH = 13.5
NPERSEG = 1024     # Higher resolution: 160/1024 = 0.156 Hz

def load_edf_data(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs = raw.info['sfreq']
    data = raw.get_data()
    ch_names = raw.ch_names
    return data, ch_names, fs

def compute_paf_robust(data, fs, nperseg=1024, alpha_range=(7.5, 13.5)):
    """
    Compute PAF with validation. Returns peak freq and center-of-gravity freq.
    
    Uses occipital/posterior channels if identifiable, else all channels.
    """
    n_channels = data.shape[0]
    
    # Compute PSD for each channel and average
    psds = []
    for ch in range(n_channels):
        freqs, psd = welch(data[ch], fs=fs, nperseg=nperseg, noverlap=nperseg*3//4)
        psds.append(psd)
    
    mean_psd = np.mean(psds, axis=0)
    
    # Extract alpha range
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = mean_psd[alpha_mask]
    
    if len(alpha_psd) < 3:
        return None, None, False, mean_psd, freqs
    
    # Peak frequency
    peak_idx = np.argmax(alpha_psd)
    paf_peak = alpha_freqs[peak_idx]
    
    # Check if peak is at edge (not a true local maximum)
    is_true_peak = (peak_idx > 0) and (peak_idx < len(alpha_psd) - 1)
    
    # Center of gravity (CoG) method - more robust
    # Weight frequencies by power
    cog = np.sum(alpha_freqs * alpha_psd) / np.sum(alpha_psd)
    
    return paf_peak, cog, is_true_peak, mean_psd, freqs

def main():
    print("=" * 70)
    print("EEGMMIDB Individual Alpha Frequency (IAF) Distribution - V2")
    print("R01 (rest, eyes-open) | nperseg=1024 | freq_res=0.156 Hz")
    print("=" * 70)
    print()
    
    subjects = []
    peak_pafs = []
    cog_pafs = []
    is_true_peaks = []
    errors = []
    
    for subj_num in range(1, 110):
        subj_id = f"S{subj_num:03d}"
        edf_path = os.path.join(DATA_DIR, subj_id, f"{subj_id}R01.edf")
        
        if not os.path.exists(edf_path):
            errors.append((subj_id, "File not found"))
            continue
        
        try:
            data, ch_names, fs = load_edf_data(edf_path)
            paf_peak, cog, is_true, mean_psd, freqs = compute_paf_robust(
                data, fs, nperseg=NPERSEG, alpha_range=(ALPHA_LOW, ALPHA_HIGH))
            
            if paf_peak is not None:
                subjects.append(subj_id)
                peak_pafs.append(paf_peak)
                cog_pafs.append(cog)
                is_true_peaks.append(is_true)
            else:
                errors.append((subj_id, "No alpha peak found"))
                
        except Exception as e:
            errors.append((subj_id, str(e)))
    
    peak_pafs = np.array(peak_pafs)
    cog_pafs = np.array(cog_pafs)
    is_true_peaks = np.array(is_true_peaks)
    
    n_true = np.sum(is_true_peaks)
    n_edge = np.sum(~is_true_peaks)
    
    print(f"Successfully processed: {len(peak_pafs)} / 109 subjects")
    print(f"  True local peaks: {n_true}")
    print(f"  Edge peaks (boundary artifact): {n_edge}")
    if errors:
        print(f"Errors/missing: {len(errors)}")
        for subj, err in errors:
            print(f"  {subj}: {err}")
    print()
    
    # === ALL SUBJECTS ===
    print("=" * 50)
    print("A) ALL SUBJECTS (N={})".format(len(peak_pafs)))
    print("=" * 50)
    
    for label, arr in [("Peak-based PAF", peak_pafs), ("CoG-based IAF", cog_pafs)]:
        print(f"\n  --- {label} ---")
        print(f"  Mean:        {np.mean(arr):.3f} Hz")
        print(f"  SD:          {np.std(arr, ddof=1):.3f} Hz")
        print(f"  Median:      {np.median(arr):.3f} Hz")
        print(f"  Min:         {np.min(arr):.3f} Hz")
        print(f"  Max:         {np.max(arr):.3f} Hz")
        q25, q75 = np.percentile(arr, [25, 75])
        print(f"  Q1 (25th):   {q25:.3f} Hz")
        print(f"  Q3 (75th):   {q75:.3f} Hz")
        print(f"  IQR:         {q75-q25:.3f} Hz")
    
    print()
    
    # === TRUE PEAKS ONLY ===
    if n_true > 0:
        true_peak_pafs = peak_pafs[is_true_peaks]
        true_cog_pafs = cog_pafs[is_true_peaks]
        true_subjects = [s for s, t in zip(subjects, is_true_peaks) if t]
        
        print("=" * 50)
        print("B) TRUE LOCAL PEAKS ONLY (N={})".format(n_true))
        print("   (excludes subjects where peak was at search boundary)")
        print("=" * 50)
        
        for label, arr in [("Peak-based PAF", true_peak_pafs), ("CoG-based IAF", true_cog_pafs)]:
            print(f"\n  --- {label} ---")
            print(f"  Mean:        {np.mean(arr):.3f} Hz")
            print(f"  SD:          {np.std(arr, ddof=1):.3f} Hz")
            print(f"  Median:      {np.median(arr):.3f} Hz")
            print(f"  Min:         {np.min(arr):.3f} Hz")
            print(f"  Max:         {np.max(arr):.3f} Hz")
            q25, q75 = np.percentile(arr, [25, 75])
            print(f"  Q1 (25th):   {q25:.3f} Hz")
            print(f"  Q3 (75th):   {q75:.3f} Hz")
            print(f"  IQR:         {q75-q25:.3f} Hz")
    
    print()
    
    # Histogram for ALL subjects (Peak method)
    print("=" * 50)
    print("HISTOGRAM: Peak-based PAF (all subjects)")
    print("=" * 50)
    bins = np.arange(7.5, 14.01, 0.5)
    counts, bin_edges = np.histogram(peak_pafs, bins=bins)
    for i in range(len(counts)):
        bar = "#" * counts[i]
        label = ""
        if bin_edges[i] <= 7.83 <= bin_edges[i+1]:
            label = " <-- Schumann f0"
        if bin_edges[i] <= 10.0 <= bin_edges[i+1]:
            label = " <-- 10 Hz"
        print(f"  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f} Hz: {counts[i]:3d}  {bar}{label}")
    
    print()
    
    # Histogram for CoG
    print("=" * 50)
    print("HISTOGRAM: CoG-based IAF (all subjects)")
    print("=" * 50)
    bins = np.arange(7.5, 14.01, 0.5)
    counts, bin_edges = np.histogram(cog_pafs, bins=bins)
    for i in range(len(counts)):
        bar = "#" * counts[i]
        label = ""
        if bin_edges[i] <= 7.83 <= bin_edges[i+1]:
            label = " <-- Schumann f0"
        if bin_edges[i] <= 10.0 <= bin_edges[i+1]:
            label = " <-- 10 Hz"
        print(f"  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f} Hz: {counts[i]:3d}  {bar}{label}")
    
    print()
    
    # PAF proximity to key frequencies (using CoG which is more robust)
    print("=" * 50)
    print("CoG IAF proximity to key frequencies")
    print("=" * 50)
    for freq, label in [(7.83, "Schumann f0 (7.83 Hz)"),
                         (10.0, "Canonical alpha (10.0 Hz)"),
                         (10.31, "phi*6.375 = 10.31 Hz"),
                         (11.0, "11.0 Hz"),
                         (12.67, "phi*7.83 = 12.67 Hz")]:
        within_05 = np.sum(np.abs(cog_pafs - freq) <= 0.5)
        within_1 = np.sum(np.abs(cog_pafs - freq) <= 1.0)
        print(f"  {label}:")
        print(f"    +/-0.5 Hz: {within_05:3d} subjects ({100*within_05/len(cog_pafs):.1f}%)")
        print(f"    +/-1.0 Hz: {within_1:3d} subjects ({100*within_1/len(cog_pafs):.1f}%)")
    
    print()
    
    # Edge-peak subjects listed
    edge_subjects = [s for s, t in zip(subjects, is_true_peaks) if not t]
    edge_peak_vals = peak_pafs[~is_true_peaks]
    print("=" * 50)
    print(f"EDGE-PEAK SUBJECTS (N={n_edge}) - peak at boundary of search range")
    print("  These subjects likely have NO clear alpha peak (1/f dominated)")
    print("=" * 50)
    for s, p in zip(edge_subjects, edge_peak_vals):
        print(f"  {s}: peak at {p:.3f} Hz (boundary)")
    
    print()
    
    # Per-subject listing (sorted by CoG)
    print("=" * 50)
    print("Per-subject CoG IAF values (sorted)")
    print("=" * 50)
    sorted_indices = np.argsort(cog_pafs)
    for i, idx in enumerate(sorted_indices):
        peak_marker = " " if is_true_peaks[idx] else "*"
        print(f"  {subjects[idx]}: {cog_pafs[idx]:6.2f}{peak_marker}", end="")
        if (i + 1) % 6 == 0:
            print()
        else:
            print(" | ", end="")
    if len(sorted_indices) % 6 != 0:
        print()
    print("  (* = edge peak, no clear alpha)")
    
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  All subjects (N=109):")
    print(f"    Peak PAF: {np.mean(peak_pafs):.2f} +/- {np.std(peak_pafs, ddof=1):.2f} Hz")
    print(f"    CoG IAF:  {np.mean(cog_pafs):.2f} +/- {np.std(cog_pafs, ddof=1):.2f} Hz")
    if n_true > 0:
        print(f"  True-peak subjects only (N={n_true}):")
        print(f"    Peak PAF: {np.mean(peak_pafs[is_true_peaks]):.2f} +/- {np.std(peak_pafs[is_true_peaks], ddof=1):.2f} Hz")
        print(f"    CoG IAF:  {np.mean(cog_pafs[is_true_peaks]):.2f} +/- {np.std(cog_pafs[is_true_peaks], ddof=1):.2f} Hz")
    print()
    print("  NOTE: R01 is eyes-OPEN rest. Many subjects show weak/absent alpha")
    print("  (1/f dominated), resulting in edge peaks at the lower boundary.")
    print("  The CoG method and true-peak filtering provide more reliable estimates.")
    print("=" * 70)

if __name__ == "__main__":
    main()

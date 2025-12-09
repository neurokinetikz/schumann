# cross_frequency

## Overview

Phase-amplitude coupling (PAC), bicoherence, and waveform shape analysis.

**Module Statistics:**
- Total Functions: 30
- Public Functions: 25
- Private Functions: 5

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_crossfreq_suite_records` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `run_schumann_locked_erpac` | `(RECORDS, sr_channel, eeg_channels, ...)` | Full ERPAC: |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_bicoherence` | `(out, i_f1=0, title=None)` | Heatmap of cross-bicoherence at a fixed f1 index across f2_grid. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_schumann_bursts` | `(RECORDS, sr_channel, ...)` | Detect Schumann bursts on a reference signal by thresholding the narrowband e... |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `slice_epoch` | `(x, idx0, idx1)` | *No description* |
| `segment_fft` | `(sig, fs, nperseg, noverlap)` | Return STFT-like complex spectra array (n_seg, n_freq) using Hann windows. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs_from_records` | `(RECORDS, time_col=_DEF_TIME_COL)` | *No description* |
| `pac_tort_mi` | `(phase, amp, nbins=18)` | *No description* |
| `pac_glm_r2` | `(phase, amp)` | *No description* |
| `pac_surrogate_z` | `(phase, amp, method='mi', n=200)` | *No description* |
| `pac_comodulogram_array` | `(x, fs, phase_bands, amp_bands, ...)` | *No description* |
| `pac_event_windows_records` | `(RECORDS, ch_name, windows, ...)` | *No description* |
| `bicoherence_array` | `(x, fs, nperseg=1024, noverlap=512, ...)` | *No description* |
| `bicoherence_event_windows_records` | `(RECORDS, ch_name, windows, ...)` | *No description* |
| `waveform_shape_metrics_array` | `(x, fs, band=(4, 8), neighborhood_ms=5.0)` | *No description* |
| `sharpness` | `(idxs)` | *No description* |
| `waveform_shape_event_windows_records` | `(RECORDS, ch_name, windows, ...)` | *No description* |
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1' (will try 'EEG.O... |
| `pac_mi_phase_amp` | `(x_phase, x_amp, nbins=18)` | Tort MI: KL divergence of phase-binned amplitude from uniform. |
| `epochwise_pac_timecourse` | `(RECORDS, eeg_channels, time_col, ...)` | Build trial x time PAC(t) around onsets, averaged over channels. |
| `cluster_permutation_1d` | `(mean_tc, trials_tc, alpha=0.05, ...)` | Simple 1D cluster-based permutation along time for ERPAC curve. |
| `cross_bicoherence` | `(RECORDS, x_sr, y_eeg, z_eeg=None, ...)` | Compute cross-bicoherence b_xy(f1,f2) predicting Z at f1+f2: |
| `idx_of` | `(freq)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_butter_bandpass` | *Helper function* |
| `_phase_amp` | *Helper function* |
| `_find_channel_series` | *Helper function* |
| `_sanitize_band` | *Helper function* |
| `_sanitize_band_list` | *Helper function* |

# pac_multiplexing

## Overview

PAC Multiplexing (Theta/Alpha → Gamma) vs Schumann SAI/Overlap (fs=128)

**Module Statistics:**
- Total Functions: 21
- Public Functions: 17
- Private Functions: 4

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_pac_vs_schumann` | `(records, fused, electrodes=None, ...)` | *No description* |
| `run_ridge_pac_coupling` | `(RECORDS, fused, electrodes=None, ...)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_pac_timeseries` | `(t, pac_ts, sai)` | *No description* |
| `plot_pac_overlap_etas` | `(t, pac_ts, overlap, k_tiers=[2, 3])` | Event-triggered averages of PAC around overlap K≥tiers onsets. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_pac_timeseries` | `(X, fs, pairs, win_sec=2.0, ...)` | Compute PAC(t) per pair with sliding windows; returns time vector and PAC dict. |
| `compute_pac_ts_custom` | `(X, fs, electrodes, phase_band, ...)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `sliding_windows` | `(n, win, step)` | *No description* |
| `circ_shift` | `(arr, shift)` | *No description* |
| `pac_mi_single` | `(x_phase, x_amp, nbins=18)` | Tort-style Modulation Index: MI = (KL divergence of phase-binned amp distribu... |
| `align_series_to_pac_timebase` | `(pac_t, sch_t, sai, overlap)` | Interpolate SAI and overlap to the PAC timebase. |
| `onsets` | `(mask)` | *No description* |
| `pac_mi_single` | `(x_phase, x_amp, nbins=18)` | *No description* |
| `sliding_windows` | `(n, win, step)` | *No description* |
| `xcorr_peak` | `(y, x, max_lag)` | Return (best_lag_samples, peak_r, lags, r) for cross-correlation of y vs x wi... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_smooth` | *Helper function* |
| `_get_fs` | *Helper function* |
| `_smooth_ts` | *Helper function* |

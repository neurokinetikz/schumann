# harmonic_resonance

## Overview

Harmonic Resonance & Spectral Mode Analysis — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 11
- Public Functions: 10
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_harmonic_resonance_spectral_modes` | `(RECORDS, eeg_channels, ...)` | High-resolution spectral harmonic test + spatial mode at 7–8 Hz. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_time_col` | `(df, candidates=...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, wins)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col=None, default_fs=128.0, ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `welch_psd` | `(x, fs, nperseg_sec=4.0)` | *No description* |
| `harmonic_zscores` | `(f, p, harmonics=..., half_bw=0.6, ...)` | For each target harmonic h, compute z = (P(h) - median(side)) / MAD(side), |
| `spatial_mode_8hz` | `(X, fs, f0=7.83, half=0.6)` | X: (n_ch, T) — band-pass 7.83±half, compute covariance → PC1 variance ratio, |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

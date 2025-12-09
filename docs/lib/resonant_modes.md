# resonant_modes

## Overview

Connectome Harmonics & Resonant Mode Analysis — Simple Graphs & Validity Tests

**Module Statistics:**
- Total Functions: 15
- Public Functions: 14
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_connectome_harmonics_resonance` | `(RECORDS, eeg_channels, ...)` | Build harmonic basis (connectome W_conn if provided; else functional PLV) and... |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `plv_adj` | `(RECORDS, channels, band, windows, ...)` | *No description* |
| `laplacian_eigendecomp` | `(W, n_modes)` | Return first n_modes Laplacian eigenvalues & eigenvectors (columns). |
| `project_to_harmonics` | `(X, H)` | X: (n_ch, T), H: (n_ch, K) columns orthonormal → A: (K, T). |
| `mode_band_power` | `(A, fs, fband)` | A: (K, T) → band power per mode via band-pass + RMS. |
| `mode_welch_power` | `(A, fs, nperseg=None)` | Return (f, Pk(f)) where Pk is (K, n_f). |
| `msc_mode_to_sr` | `(A, sr, fs, harmonics, nperseg=None)` | *No description* |
| `schumann_envelope` | `(sr, fs, center_hz=7.83, half_bw=0.6)` | *No description* |
| `schumann_envelope` | `(sig, fs, center=7.83, half=0.6)` | Compatibility wrapper: accepts center/half or center_hz/half_bw. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

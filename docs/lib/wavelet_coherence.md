# wavelet_coherence

## Overview

Wavelet coherence (WTC) analysis at Schumann frequencies.

**Module Statistics:**
- Total Functions: 10
- Public Functions: 9
- Private Functions: 1

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_sr_ignition_wtc_strip` | `(RECORDS, eeg_channel, sr_channel, ...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, wins)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `msc_harmonics_table` | `(df, eeg_channels, sr_channel, wins, ...)` | *No description* |
| `cwt_linear` | `(freqs, sig, w0, N, fs=128)` | *No description* |
| `smooth` | `(A, wlen=9)` | *No description* |
| `wavelet_coherence_tf` | `(df, x_name, y_name, ...)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

# schumann_coherence

## Overview

EEG–Schumann Coherence Testing — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 13
- Public Functions: 12
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_eeg_schumann_coherence` | `(RECORDS, eeg_channels, sr_channel, ...)` | *No description* |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_coherence_at_f0` | `(xe, xs, fs, f0, half)` | Magnitude-squared coherence between signals xe and xs at target frequency f0. |
| `build_null_threshold` | `(coh, n_null=200, method='block', ...)` | Estimate a null threshold for a sliding coherence trace by resampling the |

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
| `wavelet_coherence_tf` | `(df, x_name, y_name, ...)` | *No description* |
| `cwt_linear` | `(sig)` | *No description* |
| `smooth` | `(A, wlen=9)` | *No description* |
| `sliding_coherence_f0` | `(df, eeg_channel, sr_channel, ...)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

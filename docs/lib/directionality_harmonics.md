# directionality_harmonics

## Overview

Directionality analysis at Schumann harmonic frequencies.

**Module Statistics:**
- Total Functions: 14
- Public Functions: 11
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_directionality_harmonics` | `(RECORDS, eeg_channel, sr_channel, ...)` | (1) TV-Granger/PDC per harmonic, (2) MIMO ARX over harmonic envelopes, (3) bi... |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `narrowband_pair` | `(x_eeg, x_sr, fs, f0, half)` | *No description* |
| `windows_to_samples` | `(wins, fs, N)` | *No description* |
| `concat_segments` | `(x, segs)` | *No description* |
| `fit_mvar_2d` | `(x, y, p=6)` | *No description* |
| `pdc_from_mvar` | `(A_list, fs, f_hz)` | *No description* |
| `granger_2d_refit` | `(x, y, p=6)` | *No description* |
| `bin_idx` | `(f)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_bandpass` | *Helper function* |
| `_fft_segments` | *Helper function* |
| `_bar_for` | *Helper function* |

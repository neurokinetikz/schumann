# harmonic_locking

## Overview

Harmonic locking metrics for SR fundamentals & harmonics (0.1–60 Hz)

**Module Statistics:**
- Total Functions: 16
- Public Functions: 14
- Private Functions: 2

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_locking` | `(RECORDS, eeg_channel, sr_channel, ...)` | Compute H‑PLI per harmonic (EEG vs SR), cross‑order XH‑PLI_m (EEG harmonics v... |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_H_PLI` | `(phi_eeg, phi_sr, fs, win_sec=8.0, ...)` | *No description* |
| `compute_XH_PLI` | `(phi_m_eeg, phi1_eeg, m, fs, ...)` | *No description* |
| `compute_SubH_PLI` | `(phi_s_eeg, phi1_eeg, n, fs, ...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `phase_series` | `(x, fs, f0, half)` | *No description* |
| `sliding_centers` | `(N, fs, win_sec, step_sec)` | *No description* |
| `block_bootstrap_ci` | `(x, n_boot=1000, alpha=0.05, ...)` | *No description* |
| `pli_surrogates` | `(phi_a, phi_b, centers, win, step, ...)` | *No description* |
| `win_for_f0` | `(f0, cycles=8, min_win=8.0, max_win=120.0)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_auto_savgol` | *Helper function* |
| `_resample_to` | *Helper function* |

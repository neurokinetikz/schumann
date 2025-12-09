# cross_frequency_harmonics

## Overview

Cross-frequency coupling expansions tied to Schumann harmonics (0.1–60 Hz)

**Module Statistics:**
- Total Functions: 16
- Public Functions: 15
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_cfc_harmonics` | `(RECORDS, eeg_channel, windows, ...)` | Compute CF-PLV (1:m), PAC per order (MVL & MI with within-cycle surrogates), |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `phase_at` | `(x, fs, f0, half)` | *No description* |
| `amp_envelope` | `(x, fs, f0, half)` | *No description* |
| `windows_to_samples` | `(wins, fs, N)` | *No description* |
| `concat_segments` | `(x, segs)` | *No description* |
| `cf_plv` | `(phi1, phim, m, wins_samp)` | *No description* |
| `mvl` | `(phase, amp)` | *No description* |
| `tort_mi` | `(phase, amp, nbins=18)` | *No description* |
| `pac_within_cycle_surrogate` | `(phase_slow, amp_fast, fs, n_perm=200)` | *No description* |
| `directionality_phase_to_amp` | `(phi1, amp_m, fs, lags=...)` | *No description* |
| `directionality_amp_to_phase` | `(amp_m, phi1, fs, lags=...)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_bandpass` | *Helper function* |

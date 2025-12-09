# synchrosqueeze

## Overview

Synchrosqueezed time–frequency analysis for SR fundamentals & harmonics (0.1–60 Hz)

**Module Statistics:**
- Total Functions: 11
- Public Functions: 8
- Private Functions: 3

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `ridge_in_band` | `(P, f, t, f0, bw=0.6)` | Within [f0-bw, f0+bw], take per‑time max → ridge freq & power. |
| `circular_shift` | `(a, s)` | *No description* |
| `validate_ridge` | `(p_hat, offband_ref=None, n_perm=200, ...)` | Coverage & p‑value: how often is ridge power above off‑band reference? |
| `validate_eeg_sr_coupling` | `(p_eeg, p_sr, n_perm=200, rng=None)` | Correlation between EEG & SR ridge power with circular‑shift null. |
| `ssq_sr_validate` | `(RECORDS, eeg_channel, sr_channel, ...)` | Compute synchrosqueezed (or STFT fallback) T–F for EEG & SR; extract ridges at |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ssq_cwt_TFR` | Return (t, f, power) using synchrosqueezed CWT if available;... |
| `_safe_savefig` | *Helper function* |
| `_plot_heatmap` | *Helper function* |

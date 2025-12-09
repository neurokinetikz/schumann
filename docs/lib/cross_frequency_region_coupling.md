# cross_frequency_region_coupling

## Overview

Cross-Frequency & Cross-Region Coupling — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 24
- Public Functions: 22
- Private Functions: 2

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_cfc_cross_region` | `(RECORDS, pairs, ...)` | Same orchestrator as before, but enforces a ≤limit_high_hz scan in PAC/PLV. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_time_col` | `(df, candidates=...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, wins)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `bandpass` | `(x, f1, f2, order=4)` | *No description* |
| `bandpass` | `(x, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col=None, default_fs=128.0, ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `analytic_phase_amp` | `(x, fs, f1, f2)` | *No description* |
| `pac_mi_phase_amp` | `(phase, amp, nbins=18)` | *No description* |
| `n_m_plv` | `(phi1, phi2, n=1, m=2)` | *No description* |
| `comodulogram_pair` | `(x_phase, y_amp, fs, ...)` | Phase@x vs Amp@y PAC with an explicit upper clamp for fast frequencies. |
| `analytic_phase_amp` | `(x, f1, f2)` | *No description* |
| `nm_plv_grid` | `(x1, x2, fs, f1_range=(4, 15), ...)` | Phase locking grid with an explicit upper clamp for f2 frequencies. |
| `analytic_phase` | `(x, f1, f2)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col=None, default_fs=128.0, ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `get_sig` | `(name, wins)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |
| `_ensure_dir` | *Helper function* |

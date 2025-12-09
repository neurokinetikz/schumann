# multiscale_entropy_and_fractal_scaling

## Overview

Multi-Scale Entropy (MSE) & Fractal Scaling (DFA) — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 15
- Public Functions: 14
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_mse_dfa_multiscale` | `(RECORDS, eeg_channels, ...)` | MSE + DFA with surrogate validation, for ignition and baseline windows. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `build_drive` | `(wins)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_time_col` | `(df, candidates=...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, wins)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col=None, default_fs=128.0, ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `coarse_grain` | `(x, scale)` | Non-overlapping average; drops remainder. |
| `sampen` | `(x, m=2, r_ratio=0.2)` | Sample Entropy (m,r) with Chebyshev metric. |
| `count_matches` | `(X, tol)` | *No description* |
| `mse_curve` | `(x, fs, max_scale_sec=5.0, m=2, ...)` | Compute MSE over integer coarse-grain scales up to max_scale_sec. |
| `dfa_alpha` | `(x, fs, min_win_sec=0.25, ...)` | Detrended Fluctuation Analysis on z-scored signal (integrated profile). |
| `phase_randomize` | `(x)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

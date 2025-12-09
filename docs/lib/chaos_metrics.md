# chaos_metrics

## Overview

Recurrence Quantification & Chaos Metrics — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 18
- Public Functions: 17
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_rqa_chaos_metrics` | `(RECORDS, eeg_channels, ...)` | RQA + Chaos metrics with surrogate validation for ignition & baseline windows. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `estimate_delay_tau` | `(x, fs, max_lag_sec=2.0, method='acf-1e')` | *No description* |
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
| `takens_embedding` | `(x, m, tau)` | *No description* |
| `false_nearest_neighbors` | `(x, tau, m_list, theiler=10)` | *No description* |
| `lyapunov_rosenstein` | `(x, m, tau, fs, theiler=10, t_fit=(1, 30))` | *No description* |
| `correlation_dimension_gp` | `(X, r_min_quant=0.05, ...)` | *No description* |
| `recurrence_matrix` | `(X, eps, theiler=0)` | *No description* |
| `rqa_metrics` | `(R, lmin=2, vmin=2)` | *No description* |
| `phase_randomize` | `(x)` | *No description* |
| `pval` | `(obs, arr, greater=True)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

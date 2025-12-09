# attractor_geometry

## Overview

Topological Data Analysis of Attractor Geometry — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 16
- Public Functions: 15
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_tda_attractor_topology` | `(RECORDS, eeg_channels, ...)` | Takens embedding + persistent homology (with surrogates) to test torus-like t... |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `estimate_delay_tau` | `(x, fs, max_lag_sec=2.0, method='acf-1e')` | *No description* |

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
| `phase_randomize` | `(x)` | *No description* |
| `recurrence_plot` | `(X, eps_quant=0.1)` | *No description* |
| `max_persistence` | `(dgm)` | *No description* |
| `maxP` | `(dgm)` | *No description* |
| `count_sig` | `(dgm, thr)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

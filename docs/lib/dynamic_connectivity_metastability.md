# dynamic_connectivity_metastability

## Overview

Dynamic Connectivity & Metastability — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 18
- Public Functions: 17
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_dynamic_connectivity_metastability` | `(RECORDS, eeg_channels, ...)` | Dynamic connectivity & metastability with simple graphs and tests. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `build_surrogate_matrix` | `(X)` | *No description* |
| `build_state_matrix` | `(wins)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_time_col` | `(df, candidates=...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, wins)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_timestamp_column` | `(df, time_col=None, default_fs=128.0, ...)` | *No description* |
| `infer_fs` | `(df, time_col)` | *No description* |
| `get_series` | `(df, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `analytic_phase` | `(x, fs, f1, f2)` | *No description* |
| `pli_window` | `(Xb)` | PLI on analytic phases; Xb: (n_ch, W) bandpassed. |
| `imagcoh_window` | `(Xb)` | Imag coherency; Xb: (n_ch, W) bandpassed. |
| `sliding_windows` | `(X, fs, win_sec, step_sec)` | *No description* |
| `kuramoto_R` | `(Xb)` | *No description* |
| `phase_randomize` | `(x)` | *No description* |
| `dyn_conn_for_state` | `(X, names, label)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

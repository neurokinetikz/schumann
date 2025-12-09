# attractor_topology

## Overview

Attractor Topology via Nonlinear Dimensional Embedding — Simple Graphs & Validity Tests

**Module Statistics:**
- Total Functions: 19
- Public Functions: 18
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_attractor_topology` | `(RECORDS, eeg_channels, ...)` | Build attractor embeddings and tests for Ignition and Baseline. |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `estimate_delay_tau` | `(x, fs, max_lag_sec=2.0, method='acf-1e')` | Pick τ from the first time where autocorrelation falls below 1/e (default) or... |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_time_col` | `(RECORDS, candidates=...)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `ensure_timestamp_column` | `(RECORDS, time_col=None, ...)` | Ensure RECORDS[out_name] exists as numeric seconds (t=0 at first sample). |
| `takens_embedding` | `(x, m, tau)` | Return (N_eff, m) embedded matrix: [x_t, x_{t+τ}, ..., x_{t+(m-1)τ}] |
| `false_nearest_neighbors` | `(x, tau, m_list, theiler=10)` | Simple FNN percentage vs m. If sklearn is present, uses KDTree; else brute fo... |
| `correlation_dimension_gp` | `(X, r_min_quant=0.05, ...)` | Grassberger-Procaccia correlation sum C(r) and slope (D2) over a mid-range. |
| `lyapunov_rosenstein` | `(x, m, tau, fs, theiler=10, t_fit=(1, 30))` | Largest Lyapunov exponent (Rosenstein et al.). |
| `recurrence_plot` | `(X, eps_quant=0.1)` | Binary RP thresholded at eps = quantile(eps_quant) of distances. |
| `persistent_homology_summary` | `(X, maxdim=2)` | If ripser is installed, compute persistence and return diagrams + simple counts. |
| `count_persistent` | `(dgm, thr=0.02)` | *No description* |
| `phase_randomize` | `(x)` | *No description* |
| `time_shuffle` | `(x)` | *No description* |
| `metric_vs_surrogates` | `(metric_func, x, n_surr=100, kind='phase')` | Compute metric on x, build null from surrogates (phase or shuffle). Return (v... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

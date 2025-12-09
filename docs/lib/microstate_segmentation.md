# microstate_segmentation

## Overview

EEG Microstate Segmentation — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 18
- Public Functions: 17
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_microstate_segmentation` | `(RECORDS, eeg_channels, ...)` | Microstate maps & metrics with surrogate validation; Ignition/Baseline compar... |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `build_X` | `(wins)` | *No description* |

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
| `gfp` | `(X)` | Global Field Power across channels per time (std). X: (n_ch, T) |
| `pick_gfp_peaks` | `(G, skip)` | Pick timepoints at local maxima of GFP with a refractory 'skip' in samples. |
| `normalize_maps` | `(X)` | Zero-mean & L2-normalize topographies columnwise. X: (n_ch, Nmaps) |
| `kmeans_microstates` | `(Xmaps, k, n_init=20, seed=0)` | KMeans on normalized maps (channels×N) → centers (channels×k), labels (N,) |
| `backfit_sequence` | `(X, centers)` | Assign each timepoint to the map with max \|corr\| (polarity-invariant). |
| `smooth_labels` | `(labels, fs, min_dur_ms=30.0)` | Enforce minimum segment duration by merging short runs into neighbors. |
| `microstate_metrics` | `(labels, fs, k)` | Mean duration (ms), coverage, occurrence rate (/s), transition matrix, sequen... |
| `gev_score` | `(GFP, corr_abs)` | Global Explained Variance: sum(GFP^2 * corr^2)/sum(GFP^2). |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

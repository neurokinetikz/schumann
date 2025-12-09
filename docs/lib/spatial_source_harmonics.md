# spatial_source_harmonics

## Overview

Spatial & Source-level Harmonics (0.1–60 Hz)

**Module Statistics:**
- Total Functions: 14
- Public Functions: 12
- Private Functions: 2

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_spatial_and_source` | `(RECORDS, sr_channel, windows, ...)` | Run spatial topographies (H‑PLI, HCS), GFS bars, (optional) PLV networks and ... |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `detect_eeg_channels` | `(df, prefix='EEG.')` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `phase_series` | `(x, fs, f0, half)` | *No description* |
| `hpli_topography` | `(RECORDS, sr_channel, harmonics=..., ...)` | Compute per‑electrode H‑PLI_k maps + p‑values for each harmonic, and HCS map. |
| `wins_to_seg` | `(wins, N)` | *No description* |
| `plv_networks` | `(RECORDS, channels=None, ...)` | Compute PLV networks at each harmonic; save adjacency heatmaps & graph plots;... |
| `wins_mask` | `(N)` | *No description* |
| `lcmv_sources_at_lines` | `(RECORDS, eeg_channels, sr=None, ...)` | If MNE forward model & noise covariance are provided, compute LCMV source pow... |
| `connectome_overlap` | `(source_vec, SC, k_modes=10)` | Project a source map (N nodes) onto Laplacian eigenmodes of SC; return varian... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_bandpass` | *Helper function* |
| `_plot_topo` | *Helper function* |

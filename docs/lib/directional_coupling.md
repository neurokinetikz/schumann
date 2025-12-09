# directional_coupling

## Overview

Directional Coupling — dPLI/Granger Right‑DLPFC → Sensory (fs=128)

**Module Statistics:**
- Total Functions: 11
- Public Functions: 8
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_directional_coupling_rdlfpc_sensory` | `(RECORDS, ignition_windows, ...)` | *No description* |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_directional_deltas` | `(df)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_windows_idx` | `(t, fs, windows)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `default_clusters` | `()` | Default sensor clusters (10–20ish). Feel free to override. |
| `cluster_signal` | `(RECORDS, time_col, names)` | *No description* |
| `dpli_block` | `(x_src, x_tgt, fs, f1, f2)` | Directed PLI (src→tgt) in [f1,f2]. Returns fraction in [0,1], 0.5 ~ no direct... |
| `granger_block` | `(x_src, x_tgt, order=10)` | Pairwise GC advantage: F(src→tgt) − F(tgt→src). Requires statsmodels; else re... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_find_series` | *Helper function* |

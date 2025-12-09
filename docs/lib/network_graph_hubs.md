# network_graph_hubs

## Overview

Network Graph Metrics & Hub Analysis — Simple Graphs & Validation

**Module Statistics:**
- Total Functions: 20
- Public Functions: 19
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_graph_metrics_hubs` | `(RECORDS, eeg_channels, ...)` | Build functional graphs per band and state; compute small-worldness, clustering, |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_connectivity` | `(RECORDS, channels, wins, band, ...)` | Returns (W, chan_names, fs) — symmetric connectivity matrix in [0,1]. |

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
| `pli_connectivity` | `(X, fs, f1, f2)` | PLI from band-passed analytic phases. X: (n_ch, T) |
| `imagcoh_connectivity` | `(X, fs, f1, f2)` | Imag coherency from analytic signals. Robust to zero-lag. |
| `threshold_by_density` | `(W, density=0.2)` | Keep top fraction of weights (upper triangle) to reach target density. |
| `graph_from_weighted` | `(W)` | *No description* |
| `global_efficiency_weighted` | `(G)` | Weighted global efficiency: mean of 1/d_ij on finite shortest paths (length a... |
| `char_path_length_weighted` | `(G)` | Weighted characteristic path length using 'length' as distance. |
| `clustering_weighted` | `(G)` | *No description* |
| `modularity_greedy` | `(G)` | Greedy modularity communities (weighted). Returns membership dict and Q. |
| `participation_coeff` | `(W, memb)` | Weighted participation coefficient: 1 - sum_s (k_is/k_i)^2 |
| `small_world_sigma` | `(G, n_rewire=20)` | Small-world index σ = (C/C_rand)/(L/L_rand) using degree-preserving nulls. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

# informational_geometry

## Overview

Informational Geometry of EEG State Manifolds — Simple Graphs & Validity Tests

**Module Statistics:**
- Total Functions: 18
- Public Functions: 17
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_info_geometry_state_manifolds` | `(RECORDS, eeg_channels, ...)` | Build state vectors → embeddings → info-geom metrics, with simple graphs + te... |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |
| `slice_windows` | `(RECORDS, time_col, fs, win_sec, step_sec)` | Return list of index windows (s,e, t_center). |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `schumann_envelope` | `(sr, fs, center=7.83, half_bw=0.6)` | *No description* |
| `in_any_window` | `(t, windows)` | *No description* |
| `plv_graph_entropy` | `(RECORDS, eeg_channels, fs, s, e, band)` | Build PLV adjacency on [s:e] and return Laplacian spectral entropy. |
| `window_features` | `(RECORDS, eeg_channels, sr_channel, ...)` | Build a feature vector per sliding window: |
| `embed_states` | `(F, method='pca', n_neighbors=8, ...)` | Embed feature matrix (T×D) to 2D/3D. Returns {'X':coords, 'method':..., 'comp... |
| `trust_continuity` | `(D_high, D_low, k=8)` | Trustworthiness & Continuity (Tenenbaum / van der Maaten). |
| `ranks` | `(D)` | *No description* |
| `geodesic_stress` | `(X_high, X_low, k=8)` | Geodesic stress between k-NN geodesics (from high-D feature space) and Euclid... |
| `curvature_proxy` | `(X_high, X_low, k=8)` | Mean relative geodesic stretch over k-NN: mean_i mean_{j in N_i} (d_geo - d_e... |
| `entropy_2d_embed` | `(Z, bins=40)` | Entropy of the embedded distribution via 2D histogram (Shannon, base e). |
| `logeuclidean_spread` | `(cov_list)` | Spread on SPD manifold (Log-Euclidean): mean pairwise \|\|log(S_i) − log(S_j)\|\|_F. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

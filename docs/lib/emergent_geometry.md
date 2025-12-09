# emergent_geometry

## Overview

Phase Metric Embedding → Emergent Geometry (fs=128)

**Module Statistics:**
- Total Functions: 16
- Public Functions: 9
- Private Functions: 7

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_phase_embedding_emergent_geometry` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `analyze_state` | `(blocks, do_surrogates)` | Build PLV distance D, embed to low-D (Isomap/UMAP), and compute quality metrics. |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_phase_embedding_quality` | `(res)` | *No description* |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(records, ch_name)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `plv_distance_matrix` | `(X, fs, band)` | Return D = 1 − \|PLV\| for band-limited analytic phases across channels. |
| `trustworthiness_continuity` | `(D_high, D_low, k=5)` | Compute trustworthiness & continuity from distance matrices. |
| `geodesic_stress` | `(D_high, X_low, k=5)` | Compute normalized stress between high‑D geodesic distances and low‑D Euclide... |
| `embed_distance_matrix` | `(D, method='isomap', n_neighbors=6, ...)` | *No description* |
| `embed_distance_matrix` | `(D, method='isomap', n_neighbors=6, ...)` | Embed a precomputed distance matrix D. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_get_fs` | *Helper function* |
| `_autoelectrodes` | *Helper function* |
| `_bandpass` | *Helper function* |
| `_slice_blocks` | *Helper function* |
| `_fourier_phase_randomize_1d` | *Helper function* |
| `_make_surrogate_concat` | *Helper function* |
| `_rank_matrix` | Return rank positions per row (argsort of distances). |

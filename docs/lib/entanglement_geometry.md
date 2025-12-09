# entanglement_geometry

## Overview

Entanglement–Geometry Analogy — Plotter (fs=128)

**Module Statistics:**
- Total Functions: 13
- Public Functions: 4
- Private Functions: 9

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_entanglement_geometry_minCut_PLV` | `(x_base, x_ign, fs, bands)` | Compute per-band network metrics linking *entanglement-geometry* and synchrony: |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_entanglement_geometry_deltas` | `(df, by_session=False, ...)` | Bar plot of deltas per band. If by_session=True, aggregate by band and use me... |
| `plot_entanglement_geometry_levels` | `(df)` | Side-by-side bars for ignition vs baseline values (min-cut, entropy, PLV) per... |
| `plot_entanglement_geometry_scatter` | `(df)` | Scatter plots to visualize relationships across bands: Δmincut vs ΔPLV, Δentr... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_err_sem` | *Helper function* |
| `_butter_bandpass` | Zero-phase bandpass filter using Butterworth + filtfilt. |
| `_analytic_hilbert` | *Helper function* |
| `_compute_plv_matrix` | Compute pairwise PLV from complex analytic signals. |
| `_ensure_symmetric` | *Helper function* |
| `_laplacian_entropy` | Graph von Neumann/Laplacian entropy: |
| `_global_min_cut_weight` | Global minimum s-t cut weight using Stoer-Wagner algorithm o... |
| `_mean_upper_triangle` | *Helper function* |
| `_plv_from_timeseries` | Compute PLV matrix for a single band from real-valued time s... |

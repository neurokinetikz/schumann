# network_geometry

## Overview

State-space embeddings using UMAP and graph-theoretic network analysis.

**Module Statistics:**
- Total Functions: 19
- Public Functions: 16
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_network_geometry_suite_records` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `run_state_space_embedding_records` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `run_multi_band_geometry_records` | `(RECORDS, ignition_windows, ...)` | Run network geometry over multiple frequency bands and return a tidy DataFrame. |
| `run_full_session_with_bands_and_exports` | `(RECORDS, ignition_windows, ...)` | Run single‑band report, multi‑band table, embedding PNG, and write CSV/JSON. |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `animate_embedding_over_time_records` | `(RECORDS, window_sec, step_sec, ...)` | Create an embedding over sliding windows and save frames (and optional GIF). |
| `animate_embedding_over_time_records` | `(RECORDS, window_sec, step_sec, ...)` | *No description* |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_wpli_records` | `(X, sf, fmin, fmax)` | wPLI on (n_channels, n_times). Multitaper via mne_connectivity if available. |

## Detection

| Function | Parameters | Description |
|----------|------------|-------------|
| `find_channel_series` | `(RECORDS, ch_name)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `bandpass_guard` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs_from_records` | `(RECORDS, time_col=_DEF_TIME_COL)` | *No description* |
| `graph_entropy` | `(adj)` | *No description* |
| `minimal_cut_weight` | `(adj)` | *No description* |
| `connectome_harmonics` | `(adj, n_modes=10)` | *No description* |
| `session_report_records` | `(RECORDS, ignition_windows, ...)` | *No description* |
| `save_session_report_csv_json` | `(report, base_path)` | Write report CSV and JSON to `base_path` (without extension). Returns paths. |
| `save_embedding_plot` | `(emb, filename)` | Save a 2D/3D scatter plot of embeddings returned by run_state_space_embedding... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_discover_electrodes` | *Helper function* |
| `_slice_windows` | *Helper function* |
| `_slice` | *Helper function* |

# network_coupling

## Overview

Network-level coupling — simple validity tests & graphs

**Module Statistics:**
- Total Functions: 12
- Public Functions: 12
- Private Functions: 0

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'. |
| `plv_matrix` | `(RECORDS, channels, band, ...)` | Pairwise PLV matrix (N×N) within 'band'. Uses analytic phases from Hilbert |
| `msc_vs_sr` | `(RECORDS, channels, sr_channel, ...)` | Per-channel magnitude-squared coherence with SR at harmonics |
| `laplacian_entropy` | `(adj)` | Shannon entropy of positive Laplacian eigenvalues (normalized). |
| `global_mincut` | `(adj)` | Stoer–Wagner global min-cut on weighted undirected graph. |
| `cross_domain_graph_alignment` | `(RECORDS, eeg_channels, sr_channel, ...)` | Build EEG PLV graph in 'band'. Weight edges by geometric mean of the nodes' |
| `symmetric_orthogonalize` | `(ts)` | Symmetric orthogonalization (Colclough et al., 2015) for leakage reduction. |
| `roi_time_series` | `(RECORDS, roi_map, windows, ...)` | Build (n_roi, T) ROI matrix by averaging available channels per ROI, |
| `roi_plv_msc_vs_sr` | `(RECORDS, roi_map, sr_channel, ...)` | ROI-level PLV (phase_band) and MSC (harmonics) vs SR with circular-shift surr... |

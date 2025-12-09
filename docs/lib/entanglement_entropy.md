# entanglement_entropy

## Overview

Entanglement Entropy Analogs & Integrative Information — Simple Graphs & Tests

**Module Statistics:**
- Total Functions: 16
- Public Functions: 15
- Private Functions: 1

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_integration_analogs` | `(RECORDS, eeg_channels, band=(8, 13), ...)` | Compute integration/complexity measures and simple tests; produce figures + C... |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_state` | `(wins, state_name)` | *No description* |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | *No description* |
| `bandpass` | `(x, fs, f1, f2, order=4)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | *No description* |
| `zscore` | `(x)` | *No description* |
| `plv_matrix` | `(RECORDS, channels, band, windows, ...)` | *No description* |
| `laplacian_spectral_entropy` | `(A)` | *No description* |
| `gaussian_entropies` | `(X)` | X: (n_ch, T) z-scored |
| `pca_first_component` | `(X)` | *No description* |
| `lz_complexity_binary` | `(seq)` | LZ76 complexity of a binary sequence (0/1), normalized by n/log2(n). |
| `permutation_entropy` | `(x, m=3, tau=1)` | Band-limited x; simple permutation entropy of order m (permutation count m! b... |
| `circular_shift_null` | `(xmat, n_surr=200)` | Circularly shift each channel independently; returns list of surrogates (n_ch... |
| `pval` | `(obs, null)` | *No description* |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |

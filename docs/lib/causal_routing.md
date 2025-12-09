# causal_routing

## Overview

Directed Connectivity & Causal Routing — Simple Graphs and Validation Tests

**Module Statistics:**
- Total Functions: 24
- Public Functions: 22
- Private Functions: 2

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_directed_connectivity_routing` | `(RECORDS, eeg_channels, ...)` | Map directed connectivity inside the brain and between brain and field (SR). |

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
| `pick_posterior_sr` | `(RECORDS)` | *No description* |
| `make_roi_series` | `(RECORDS, eeg_channels, roi_map=None, ...)` | Return (X, names, fs) where X is (n_nodes, T) z-scored per node. |
| `fit_var_robust` | `(X, fs, order_max=16, trend='n', ...)` | Robust VAR(p) fit with manual BIC selection and PD enforcement on Σ_u. |
| `A_of_f` | `(A, f, fs)` | A(f) = I − Σ_k A_k e^{−i2πfk/fs};  returns (n_f, n, n). |
| `H_of_f` | `(Af)` | Transfer matrix H(f) = A(f)^{-1}, per frequency. |
| `spectral_dtf_pdc` | `(A, fs, fmin, fmax, n_freq=128)` | DTF_{i<-j}(f) = \|H_ij\| / sqrt(Σ_k \|H_ik\|^2)   (row-normalized) |
| `band_average` | `(M, f, band)` | Mean over frequency indices in band; M shape (n,n,n_f). |
| `fit_var` | `(X, order_max=20, criterion='bic')` | Fit VAR to (n_nodes, T) → statsmodels VAR on (T, n_nodes). |
| `A_of_f` | `(A, f, fs)` | A(f) = I − Σ_k A_k e^{−i2πfk/fs};  returns (n_f, n, n). |
| `H_of_f` | `(Af)` | Transfer matrix H(f) = A(f)^{-1}, per frequency. |
| `spectral_dtf_pdc` | `(A, fs, fmin, fmax, n_freq=128)` | DTF_{i<-j}(f) = \|H_ij\| / sqrt(Σ_k \|H_ik\|^2)   (row-normalized) |
| `band_average` | `(M, f, band)` | Mean over frequency indices in band; M shape (n,n,n_f). |
| `circular_shift_sr_null` | `(RECORDS, sr_channel, fs, windows, ...)` | Build null distribution for SR→ROI DTF at ~f0 by circularly shifting SR and r... |
| `pca_reduce_nodes` | `(X, k)` | X: (n_nodes, T) z-scored. Returns (Z, U) where |
| `granger_bivariate_matrix` | `(X, maxlag=6)` | X: (n_nodes, T) z-scored. |
| `fit_var_safeguarded` | `(X, fs, order_max=16, ridge=1e-08)` | Adaptive lag cap from data length; manual BIC; PD enforcement on Sigma_u. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_ensure_dir` | *Helper function* |
| `_align` | *Helper function* |

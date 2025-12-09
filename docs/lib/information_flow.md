# information_flow

## Overview

Directionality & Information Flow (stand-alone)

4a) Frequency-domain Granger / Partial Directed Coherence (PDC) / DTF
    • Fit VAR (bivariate or multivariate) on EEG + Schumann reference
    • Diagnostics: order selection (AIC/BIC), stability (roots<1), residual whiteness (Ljung-Box)
    • Spectral DTF/PDC and (optional) time-domain Granger tests
    • Report values at Schumann harmonics (≈7.83, 14.3, 20.8, 27.3, 33.8 Hz)

4b) Transfer Entropy (TE) / Conditional TE (lag-resolved)

**Module Statistics:**
- Total Functions: 25
- Public Functions: 17
- Private Functions: 8

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `run_freq_granger_pdc_dtf` | `(RECORDS, channels, windows=None, ...)` | Fit VAR, compute DTF/PDC spectra, and report harmonic values. |
| `run_transfer_entropy` | `(RECORDS, x_channel, y_channel, ...)` | Compute TE(X→Y) and TE(Y→X) across lags (ms). Returns arrays and surrogate 95... |
| `run_tvar_dtf` | `(RECORDS, channels, windows=None, ...)` | Time-varying AR (Kalman-RLS) and DTF(t, i<-j) at f0. |

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_dtf_grid_bidir_like_single` | `(tv_ign, tv_base, src_channel, ...)` | Grid of DTF(t, target <- source) for all targets (all electrodes except source). |

## Data Processing

| Function | Parameters | Description |
|----------|------------|-------------|
| `slice_concat` | `(x, fs, windows)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `infer_fs` | `(RECORDS, time_col='Timestamp')` | *No description* |
| `get_series` | `(RECORDS, name)` | Return numeric signal. Accepts 'EEG.O1' or bare 'O1' (tries 'EEG.O1'). |
| `stack_channels` | `(RECORDS, channels, fs, windows, ...)` | *No description* |
| `fit_var_model` | `(X, order_max=20, crit='bic')` | Fit VAR to (n_ch, L) array X.T using statsmodels. |
| `spectral_dtf_pdc` | `(A, Sigma_u, fs, fmin=0.0, fmax=50.0, ...)` | Compute DTF and PDC spectra from VAR(A, Sigma_u). |
| `summarize_dtf_pdc_at_harmonics` | `(spec, harm)` | *No description* |
| `transfer_entropy_knn` | `(x, y, lag, k_embed_x=1, k_embed_y=1, k=4)` | TE X→Y at a given *positive* lag (samples): predicts y_{t+lag} from [y_t^(k_e... |
| `embed` | `(sig, kdim)` | *No description* |
| `kalman_rls_tvar_ar` | `(X, order=4, lam=0.995)` | Track time-varying AR coefficients for a multivariate series X (n_ch, L) |
| `phi_t` | `(t_idx)` | *No description* |
| `dtf_at_freq_from_A` | `(A, fs, f0)` | DTF at a single frequency f0 from AR matrices A (p,n,n). |
| `transfer_entropy_knn` | `(x, y, lag, k_embed_x=1, k_embed_y=1, k=4)` | TE X→Y at positive sample lag. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_lb_last_pvalue` | *Helper function* |
| `_A_of_f` | A(f) = I - sum_{k=1..p} A_k e^{-i 2π f k / fs}, shape: (n_fr... |
| `_H_of_f` | Transfer matrix H(f) = A(f)^{-1}, for each frequency. Af sha... |
| `_knn_entropy` | Shannon differential entropy via Kozachenko–Leonenko (Euclid... |
| `_knn_entropy` | Differential entropy via Kozachenko–Leonenko (Euclidean). |
| `_embed` | Simple delay embedding with unit delay: [x_t, x_{t+1}, ..., ... |
| `_moving_average` | *Helper function* |
| `_mean_in_seconds` | *Helper function* |

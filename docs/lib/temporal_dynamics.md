# temporal_dynamics

## Overview

Lead/Lag Quantification Among SR Families (Temporal Dynamics)

**Module Statistics:**
- Total Functions: 19
- Public Functions: 16
- Private Functions: 3

## Main Analysis

| Function | Parameters | Description |
|----------|------------|-------------|
| `analyze_lead_lag_temporal` | `(RECORDS, eeg_channel, windows, ...)` | Compute envelope xcorr lags & envelope‑phase lead probabilities per window an... |
| `analyze_window` | `(win_name, segs)` | *No description* |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `ensure_dir` | `(d)` | *No description* |
| `ensure_timestamp_column` | `(df, time_col='Timestamp', ...)` | *No description* |
| `infer_fs` | `(df, time_col='Timestamp')` | *No description* |
| `band_envelope_and_slow_phase` | `(x, fs, f0, half=0.6, ...)` | Return: slow_env (band‑passed envelope in slow_band), slow_phase (Hilbert ang... |
| `windows_to_samples` | `(wins, fs, N)` | *No description* |
| `block_bootstrap_series` | `(x, block_N, out_N, rng)` | *No description* |
| `xcorr_lag` | `(env_f, env_ref, fs, max_lag_s=30.0)` | *No description* |
| `lag_ci_bootstrap` | `(env_f, env_ref, fs, max_lag_s=30.0, ...)` | *No description* |
| `phase_lead_probability` | `(phi_f, phi_ref, n_boot=1000, ...)` | Return P_lead = Pr(Δϕ ∈ (0,π)) with block‑bootstrap 95% CI. |
| `env_phase_for` | `(f)` | *No description* |
| `consensus_order_score` | `(fam_lags, tol=2.0)` | Return fraction of pairwise constraints satisfied for ordering: |
| `inc` | `(test)` | *No description* |
| `virtual_eeg_snr_weighted` | `(RECORDS, channels, fs, f0, half=0.6, ...)` | Return (v_sig, weights) where v_sig is SNR‑weighted sum of channels normalize... |
| `snr_at_f0` | `(x, fs, f0, half=0.6, flank=2.0)` | SNR = band power at f0±half / average flank power at [f0±(half+δ) .. f0±(half... |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_butter_bandpass` | *Helper function* |
| `_butter_lowpass` | *Helper function* |
| `_bandpass` | *Helper function* |

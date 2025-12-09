# temporal_dynamics

**Total functions:** 19 (16 public, 3 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `band_envelope_and_slow_phase(x, fs, f0, half, slow_band)`
> Return: slow_env (band‑passed envelope in slow_band), slow_phase (Hilbert angle).

### `windows_to_samples(wins, fs, N)`

### `block_bootstrap_series(x, block_N, out_N, rng)`

### `xcorr_lag(env_f, env_ref, fs, max_lag_s)`

### `lag_ci_bootstrap(env_f, env_ref, fs, max_lag_s, n_boot, ...)`

### `phase_lead_probability(phi_f, phi_ref, n_boot, block_len, seed)`
> Return P_lead = Pr(Δϕ ∈ (0,π)) with block‑bootstrap 95% CI.

### `analyze_lead_lag_temporal(RECORDS, eeg_channel: str, windows: dict, time_col, fundamental, ...)`
> Compute envelope xcorr lags & envelope‑phase lead probabilities per window and family.

### `env_phase_for(f)`

### `analyze_window(win_name, segs)`

### `consensus_order_score(fam_lags, tol)`
> Return fraction of pairwise constraints satisfied for ordering:

### `inc(test)`

### `virtual_eeg_snr_weighted(RECORDS, channels, fs, f0, half, ...)`
> Return (v_sig, weights) where v_sig is SNR‑weighted sum of channels normalized to unit gain.

### `snr_at_f0(x, fs, f0, half, flank)`
> SNR = band power at f0±half / average flank power at [f0±(half+δ) .. f0±(half+δ+flank)].

## Private/Helper Functions

- `_butter_bandpass(x, fs, lo, ...)`
- `_butter_lowpass(x, fs, hi, ...)`
- `_bandpass(x, fs, lo, ...)`

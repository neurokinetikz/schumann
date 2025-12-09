# synchrosqueeze

**Total functions:** 11 (8 public, 3 private)

## Public Functions

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `get_series(df, name)`

### `ridge_in_band(P, f, t, f0, bw)`
> Within [f0-bw, f0+bw], take per‑time max → ridge freq & power.

### `circular_shift(a, s)`

### `validate_ridge(p_hat, offband_ref, n_perm, rng)`
> Coverage & p‑value: how often is ridge power above off‑band reference?

### `validate_eeg_sr_coupling(p_eeg, p_sr, n_perm, rng)`
> Correlation between EEG & SR ridge power with circular‑shift null.

### `ssq_sr_validate(RECORDS, eeg_channel: str, sr_channel: str, time_col, freq_groups, ...)`
> Compute synchrosqueezed (or STFT fallback) T–F for EEG & SR; extract ridges at

## Private/Helper Functions

- `_ssq_cwt_TFR(x, fs, fmin, ...)`
- `_safe_savefig(path, dpi)`
- `_plot_heatmap(t, f, P, ...)`

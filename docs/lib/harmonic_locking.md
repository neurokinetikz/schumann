# harmonic_locking

**Total functions:** 16 (14 public, 2 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `get_series(df, name)`

### `bandpass(x, fs, f1, f2, order)`

### `phase_series(x, fs, f0, half)`

### `sliding_centers(N, fs, win_sec, step_sec)`

### `block_bootstrap_ci(x, n_boot, alpha, block_len, seed)`

### `pli_surrogates(phi_a, phi_b, centers, win, step, ...)`

### `compute_H_PLI(phi_eeg, phi_sr, fs, win_sec, step_sec, ...)`

### `compute_XH_PLI(phi_m_eeg, phi1_eeg, m, fs, win_sec, ...)`

### `compute_SubH_PLI(phi_s_eeg, phi1_eeg, n, fs, win_sec, ...)`

### `win_for_f0(f0, cycles, min_win, max_win)`

### `analyze_locking(RECORDS, eeg_channel: str, sr_channel: str, fundamental, harmonics, ...)`
> Compute H‑PLI per harmonic (EEG vs SR), cross‑order XH‑PLI_m (EEG harmonics vs EEG fundamental),

## Private/Helper Functions

- `_auto_savgol(y, max_window)`
- `_resample_to(t_src, y_src, t_ref)`

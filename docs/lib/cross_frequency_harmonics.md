# cross_frequency_harmonics

**Total functions:** 16 (15 public, 1 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `get_series(df, name)`

### `phase_at(x, fs, f0, half)`

### `amp_envelope(x, fs, f0, half)`

### `windows_to_samples(wins, fs, N)`

### `concat_segments(x, segs)`

### `cf_plv(phi1, phim, m, wins_samp)`

### `mvl(phase, amp)`

### `tort_mi(phase, amp, nbins)`

### `pac_within_cycle_surrogate(phase_slow, amp_fast, fs, n_perm)`

### `directionality_phase_to_amp(phi1, amp_m, fs, lags)`

### `directionality_amp_to_phase(amp_m, phi1, fs, lags)`

### `analyze_cfc_harmonics(RECORDS, eeg_channel: str, windows: dict, time_col, fundamental, ...)`
> Compute CF-PLV (1:m), PAC per order (MVL & MI with within-cycle surrogates),

## Private/Helper Functions

- `_bandpass(x, fs, lo, ...)`

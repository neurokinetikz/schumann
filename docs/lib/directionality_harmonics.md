# directionality_harmonics

**Total functions:** 14 (11 public, 3 private)

## Public Functions

### `ensure_dir(d)`

### `ensure_timestamp_column(df, time_col, default_fs)`

### `infer_fs(df, time_col)`

### `narrowband_pair(x_eeg, x_sr, fs, f0, half)`

### `windows_to_samples(wins, fs, N)`

### `concat_segments(x, segs)`

### `fit_mvar_2d(x, y, p)`

### `pdc_from_mvar(A_list, fs, f_hz)`

### `granger_2d_refit(x, y, p)`

### `analyze_directionality_harmonics(RECORDS, eeg_channel: str, sr_channel: str, windows: dict, time_col, ...)`
> (1) TV-Granger/PDC per harmonic, (2) MIMO ARX over harmonic envelopes, (3) bispectral directionality.

### `bin_idx(f)`

## Private/Helper Functions

- `_bandpass(x, fs, lo, ...)`
- `_fft_segments(x, fs, nperseg, ...)`
- `_bar_for(metric_prefix)`

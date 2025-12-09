# harmonic_coherence

**Total functions:** 8 (6 public, 2 private)

## Public Functions

### `compute_coherence_at_f0(xe, xs, fs, f0, half)`
> Magnitude-squared coherence between signals xe and xs at target frequency f0.

### `sliding_coherence_f0(df, eeg_channel, sr_channel, ignition_windows, f0, ...)`

### `plot_sr_ignition_signature(records, eeg_channel: str, sr_channel: str, ignition_windows, time_col, ...)`
> Enhanced sliding-coherence plotter: clearer lines, non-overlapping labels/legend,

### `build_null_threshold(coh, n_null, method, block_len, alpha, ...)`
> Estimate a null threshold for a sliding coherence trace by resampling the

### `build_null_threshold_smooth(coh_raw, n_null, method, block_len, alpha, ...)`
> Compute a (1-alpha) null threshold compatible with *smoothed* plotting.

### `zscore_with_series(value, series, eps)`
> Convert a scalar threshold `value` into z-units using the mean/std of `series`.

## Private/Helper Functions

- `_auto_savgol(y, max_window)`
- `_clip_shading(ax, windows, tmin, ...)`

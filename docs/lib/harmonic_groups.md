# harmonic_groups

**Total functions:** 10 (8 public, 2 private)

## Public Functions

### `sr_groups()`

### `win_for_f0(f0, cycles, min_win, max_win)`
> Adaptive window: ensure ≥cycles of f0, clipped to [min_win, max_win].

### `half_bw_for_win(win_sec, mult, min_bw)`
> Choose half-bandwidth from spectral resolution Δf≈2/win_sec, scaled by mult.

### `build_null_threshold(coh, n_null, method, block_len, alpha, ...)`
> Estimate a null threshold for a sliding coherence trace by resampling the

### `compute_coherence_at_f0(xe, xs, fs, f0, half)`
> Magnitude-squared coherence between signals xe and xs at target frequency f0.

### `sliding_coherence_f0(df, eeg_channel, sr_channel, ignition_windows, f0, ...)`

### `plot_sr_group_adaptive(records, eeg_channel: str, sr_channel: str, group_name: str, ignition_windows, ...)`
> Compute & plot sliding z-coherence for all f0 in the chosen group with

### `summarize_sr_groups(records, eeg_channel, sr_channel, ignition_windows, groups)`

## Private/Helper Functions

- `_maybe_smoother()`
- `_build_null_for_series(coh_raw, n_null, smoother)`

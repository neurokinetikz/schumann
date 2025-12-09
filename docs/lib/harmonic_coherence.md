# harmonic_coherence

## Overview

EEG-Schumann coherence signatures and ignition-locked analysis.

**Module Statistics:**
- Total Functions: 8
- Public Functions: 6
- Private Functions: 2

## Plotting

| Function | Parameters | Description |
|----------|------------|-------------|
| `plot_sr_ignition_signature` | `(records, eeg_channel, sr_channel, ...)` | Enhanced sliding-coherence plotter: clearer lines, non-overlapping labels/leg... |

## Computation

| Function | Parameters | Description |
|----------|------------|-------------|
| `compute_coherence_at_f0` | `(xe, xs, fs, f0, half)` | Magnitude-squared coherence between signals xe and xs at target frequency f0. |
| `build_null_threshold` | `(coh, n_null=200, method='block', ...)` | Estimate a null threshold for a sliding coherence trace by resampling the |
| `build_null_threshold_smooth` | `(coh_raw, n_null=200, method='block', ...)` | Compute a (1-alpha) null threshold compatible with *smoothed* plotting. |

## Utilities

| Function | Parameters | Description |
|----------|------------|-------------|
| `sliding_coherence_f0` | `(df, eeg_channel, sr_channel, ...)` | *No description* |
| `zscore_with_series` | `(value, series, eps=1e-12)` | Convert a scalar threshold `value` into z-units using the mean/std of `series`. |

## Private Helpers

| Function | Description |
|----------|-------------|
| `_auto_savgol` | Light, safe smoothing for visibility (odd window; poly=2). |
| `_clip_shading` | *Helper function* |
